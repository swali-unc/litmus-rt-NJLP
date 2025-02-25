/*
 * litmus/sched_gsn_edf2.c
 *
 * This is the Earliest Deadline Zero Laxity (EDZL) scheduler implementation.
 * It is implemented as a modification of GSN_EDF.
 * @author Zelin Tong ztong@cs.unc.edu
 *
 * The idea of EDZL is simple, everything works according to global earliest
 * deadline first with one exception. First, we define laxity as the remaining
 * time before the deadline, minus the remaining worst-case execution time of the task.
 * Should the task hit zero laxity, the task's priority is boosted.
 *
 * As such, EDZL is not a job-level fixed-priority scheduler. Because EDZL also
 * has benefits over regular GSN_EDF, we test the NJLP and FMLP on the EDZL scheduler.
 * 
 * The NJLP is a locking protocol that prioritizes tasks that have experienced
 * the most pi-blocking. It has a blocking bound proportional to m, the number
 * of processors, and as such, it is the first asymptotically optimal non-JLFP
 * locking protocol.
 *
 * To facilitate prioritizing jobs by pi-blocking, we require each task to know
 * how much pi-blocking it has experienced. This results in a two-fold question,
 * how does one determine the top m jobs in a system when such a thing is not normally
 * tracked (because wait-queue'd jobs are not normally present in the ready and cpu queue)?
 * The second question is how much overhead is involved in keeping track of pi-blocking
 * anyway?
 *
 * The first answer results in doubled scheduling logic as we now need to keep track
 * of ALL jobs in the system in a single queue. The second answer requires additional
 * tracking of jobs when they are scheduled, locking, unlocking, and completing.
 *
 * When looking at the doubled scheduling logic, it is important to note the
 * difference between tracked and linked. Something that is linked is affecting the
 * actual CPU queues. Data structures and functions referencing "tracked" are related
 * to just tracking a job rather than actually scheduling it.
 *
 * Please see the paper for more information on the theory behind the NJLP. This file
 * contains an example implementation.
 */

#include <linux/spinlock.h>
#include <linux/percpu.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/sched/topology.h>
#include <linux/slab.h>
#include <linux/hrtimer.h>

#include <litmus/debug_trace.h>
#include <litmus/litmus.h>
#include <litmus/budget.h>
#include <litmus/jobs.h>
#include <litmus/sched_plugin.h>
#include <litmus/edzl_common.h>
#include <litmus/sched_trace.h>
#include <litmus/trace.h>

#include <litmus/preempt.h>
#include <litmus/budget.h>
#include <litmus/np.h>

#include <litmus/bheap.h>

#ifdef CONFIG_SCHED_CPU_AFFINITY
#include <litmus/affinity.h>
#endif

/* to set up domain/cpu mappings */
#include <litmus/litmus_proc.h>

#include <linux/module.h>

/* Overview of GSN-EDF operations.
 *
 * For a detailed explanation of GSN-EDF have a look at the FMLP paper. This
 * description only covers how the individual operations are implemented in
 * LITMUS.
 *
 * link_task_to_cpu(T, cpu) 	- Low-level operation to update the linkage
 *                                structure (NOT the actually scheduled
 *                                task). If there is another linked task To
 *                                already it will set To->linked_on = NO_CPU
 *                                (thereby removing its association with this
 *                                CPU). However, it will not requeue the
 *                                previously linked task (if any). It will set
 *                                T's state to not completed and check whether
 *                                it is already running somewhere else. If T
 *                                is scheduled somewhere else it will link
 *                                it to that CPU instead (and pull the linked
 *                                task to cpu). T may be NULL.
 *
 * unlink(T)			- Unlink removes T from all scheduler data
 *                                structures. If it is linked to some CPU it
 *                                will link NULL to that CPU. If it is
 *                                currently queued in the gsnedf queue it will
 *                                be removed from the rt_domain. It is safe to
 *                                call unlink(T) if T is not linked. T may not
 *                                be NULL.
 *
 * requeue(T)			- Requeue will insert T into the appropriate
 *                                queue. If the system is in real-time mode and
 *                                the T is released already, it will go into the
 *                                ready queue. If the system is not in
 *                                real-time mode is T, then T will go into the
 *                                release queue. If T's release time is in the
 *                                future, it will go into the release
 *                                queue. That means that T's release time/job
 *                                no/etc. has to be updated before requeu(T) is
 *                                called. It is not safe to call requeue(T)
 *                                when T is already queued. T may not be NULL.
 *
 * gsnedf_job_arrival(T)	- This is the catch all function when T enters
 *                                the system after either a suspension or at a
 *                                job release. It will queue T (which means it
 *                                is not safe to call gsnedf_job_arrival(T) if
 *                                T is already queued) and then check whether a
 *                                preemption is necessary. If a preemption is
 *                                necessary it will update the linkage
 *                                accordingly and cause scheduled to be called
 *                                (either with an IPI or need_resched). It is
 *                                safe to call gsnedf_job_arrival(T) if T's
 *                                next job has not been actually released yet
 *                                (releast time in the future). T will be put
 *                                on the release queue in that case.
 *
 * curr_job_completion()	- Take care of everything that needs to be done
 *                                to prepare the current task for its next
 *                                release and place it in the right queue with
 *                                gsnedf_job_arrival().
 *
 *
 * When we now that T is linked to CPU then link_task_to_cpu(NULL, CPU) is
 * equivalent to unlink(T). Note that if you unlink a task from a CPU none of
 * the functions will automatically propagate pending task from the ready queue
 * to a linked task. This is the job of the calling function ( by means of
 * __take_ready).
 */


/* struct for semaphore with priority inheritance */
struct njlp_semaphore {
	struct litmus_lock litmus_lock;

	/* current resource holder */
	struct task_struct *owner;

	/* highest-priority waiter */
	struct task_struct *hp_waiter;

	/* priority queue of waiting tasks */
	struct bheap waitq; // njlp_priority_order
	struct bheap waitq2; // edzl_ready_order
	spinlock_t waitlock; // The waitlock is used to work on locking data structures
	spinlock_t qlock;  // The qlock specifically locks down wait queues which affect inheritance
};

// Obtains the njlp_semaphore when only passed the litmus_lock member
static inline struct njlp_semaphore* njlp_from_lock(struct litmus_lock* lock)
{
	return container_of(lock, struct njlp_semaphore, litmus_lock);
}

/* cpu_entry_t - maintain the linked and scheduled state
 */
typedef struct  {
	int 			cpu;
	struct task_struct*	linked;		/* only RT tasks */
	struct task_struct*	scheduled;	/* only RT tasks */

	// tracked lets us track the top m jobs in the system.
	// If a job is tracked, then it is in the top m, but that
	// doesn't mean it is necessarily scheduled.
	struct task_struct* tracked;

	// The heap nodes below are according to priority order and base priority order
	// respectively. The base priority order is important for NJLP because inherited
	// priorities are not helpful when determining pi-blocking time.
	struct bheap_node*	hn;
	struct bheap_node* hn2;
} cpu_entry_t;
DEFINE_PER_CPU(cpu_entry_t, gsnedf2_cpu_entries);

// The cpu entries. NR_CPUs is the number of threads on the system
cpu_entry_t* gsnedf2_cpus[NR_CPUS];

/* the cpus queue themselves according to priority in here */
// Note, specific to the NJLP, note there is doubled scheduling logic.
static struct bheap_node gsnedf_heap_node[NR_CPUS];
static struct bheap_node gsnedf_heap_node2[NR_CPUS];
static struct bheap      gsnedf_cpu_heap;
static struct bheap      gsnedf_cpu_heap2;

// The rt_domain contains everything related to this implementation of EDZL
static rt_domain_t gsnedf;
#define gsnedf_lock (gsnedf.ready_lock)


/* Uncomment this if you want to see all scheduling decisions in the
 * TRACE() log.
#define WANT_ALL_SCHED_EVENTS
 */

// This function compares two cpu bheap nodes to determine which
// cpu job has a lower priority according to EDZL
static int cpu_lower_prio(struct bheap_node *_a, struct bheap_node *_b)
{
	cpu_entry_t *a, *b;
	a = _a->value;
	b = _b->value;
	/* Note that a and b are inverted: we want the lowest-priority CPU at
	 * the top of the heap.
	 */
	return edzl_higher_prio(b->linked, a->linked);
}

// Similar to the previous function, but with base priorities instead.
// Inherited priorities are not helpful when determining pi-blocking
static int cpu_lower_base_prio(struct bheap_node *_a, struct bheap_node *_b)
{
	cpu_entry_t *a, *b;
	a = _a->value;
	b = _b->value;
	/* Note that a and b are inverted: we want the lowest-priority CPU at
	 * the top of the heap.
	 */
	return edzl_higher_base_prio(b->linked, a->linked);
}

/* update_cpu_position - Move the cpu entry to the correct place to maintain
 *                       order in the cpu queue. Caller must hold gsnedf lock.
 */
static void update_cpu_position(cpu_entry_t *entry)
{
	if (likely(bheap_node_in_heap(entry->hn)))
		bheap_delete(cpu_lower_prio, &gsnedf_cpu_heap, entry->hn);
	bheap_insert(cpu_lower_prio, &gsnedf_cpu_heap, entry->hn);
}

// NJLP-specific version of the previous method where we are operating
// using the secondary data structures.
static void update_cpu_position2(cpu_entry_t *entry)
{
	if (likely(bheap_node_in_heap(entry->hn2)))
		bheap_delete(cpu_lower_base_prio, &gsnedf_cpu_heap2, entry->hn2);
	bheap_insert(cpu_lower_base_prio, &gsnedf_cpu_heap2, entry->hn2);
}

/* caller must hold gsnedf lock */
// If there is going to be a job on the cpu that will be preempted,
// it will be the lowest priority one first, so this function will help
// find that.
static cpu_entry_t* lowest_prio_cpu(void)
{
	struct bheap_node* hn;
	hn = bheap_peek(cpu_lower_prio, &gsnedf_cpu_heap);
	return hn->value;
}

// The NJLP version of the previous method where we use the secondary
// data structures.
static cpu_entry_t* lowest_base_prio_cpu(void)
{
	struct bheap_node* hn;
	hn = bheap_peek(cpu_lower_base_prio, &gsnedf_cpu_heap2);
	return hn->value;
}

// The NJLP's priority order is determined by whichever task has pi-blocked longer.
static int njlp_priority_order(struct bheap_node* a, struct bheap_node* b)
{
	struct task_struct* ta = bheap2task(a);
	struct task_struct* tb = bheap2task(b);

	return (tsk_rt(ta)->pi_blocked > tsk_rt(tb)->pi_blocked);
}

// This will update the pi-blocking of a task but also perform some extra
// checks.
static void try_update_pi_blocking(struct task_struct* t, int update_last)
{
	lt_t now;
	struct njlp_semaphore* sem;
	unsigned long flags;

	//  A sanity check to make sure the task is actually wait queue'd and
	// the task isn't NULL (if a core became idle, the next task is NULL,
	// so make sure that isn't passed here)
	if (!t || !is_waitqueued(t))
		return;

	sem = njlp_from_lock(tsk_rt(t)->sem);

	spin_lock(&sem->qlock);
	if (is_waitqueued(t)) {
		BUG_ON(!tsk_rt(t)->last_updated);

		/* Update pi-blocking */
		now = litmus_clock();
		tsk_rt(t)->pi_blocked += now - tsk_rt(t)->last_updated;
		if (update_last)
			tsk_rt(t)->last_updated = now;
		else
			tsk_rt(t)->last_updated = 0;

		/* Re-order waitq heap due to pi-blocking change */
		bheap_decrease(njlp_priority_order, tsk_rt(t)->waitq_heap_node);
	}
	spin_unlock(&sem->qlock);
}

/* link_task_to_cpu - Update the link of a CPU.
 *                    Handles the case where the to-be-linked task is already
 *                    scheduled on a different CPU.
 */
static noinline void link_task_to_cpu(struct task_struct* linked,
				      cpu_entry_t *entry)
{
	cpu_entry_t *sched;
	struct task_struct* tmp;
	int on_cpu;

	// Only want to deal with real-time tasks
	BUG_ON(linked && !is_realtime(linked));

	/* Currently linked task is set to be unlinked. */
	if (entry->linked) {
		entry->linked->rt_param.linked_on = NO_CPU;
	}

	/* Link new task to CPU. */
	if (linked) {
		/* handle task is already scheduled somewhere! */
		on_cpu = linked->rt_param.scheduled_on;
		if (on_cpu != NO_CPU) {
			sched = &per_cpu(gsnedf2_cpu_entries, on_cpu);
			/* this should only happen if not linked already */
			BUG_ON(sched->linked == linked);

			/* If we are already scheduled on the CPU to which we
			 * wanted to link, we don't need to do the swap --
			 * we just link ourselves to the CPU and depend on
			 * the caller to get things right.
			 */
			if (entry != sched) {
				TRACE_TASK(linked,
					   "already scheduled on %d, updating link.\n",
					   sched->cpu);
				tmp = sched->linked;
				linked->rt_param.linked_on = sched->cpu;
				sched->linked = linked;
				update_cpu_position(sched);
				linked = tmp;
			}
		}
		if (linked) /* might be NULL due to swap */
			linked->rt_param.linked_on = entry->cpu;
	}
	entry->linked = linked;
#ifdef WANT_ALL_SCHED_EVENTS
	if (linked)
		TRACE_TASK(linked, "linked to %d.\n", entry->cpu);
	else
		TRACE("NULL linked to %d.\n", entry->cpu);
#endif
	update_cpu_position(entry);
}

// This is the NJLP version of the previous function.
// Recall, tracked to a cpu does not mean you are actually scheduled on it.
// Instead, if you are tracked on a cpu, then you are in the top m priorities.
static noinline void track_task_to_cpu(struct task_struct* tracked,
				      cpu_entry_t *entry)
{
	lt_t now = litmus_clock();

	BUG_ON(tracked && !is_realtime(tracked));

	/* Currently tracked task is set to be untracked. */
	if (entry->tracked) {
		entry->tracked->rt_param.tracked_on = NO_CPU;
		try_update_pi_blocking(entry->tracked, 0);
	}

	/* Track new task to CPU. */
	if (tracked) {
		tracked->rt_param.tracked_on = entry->cpu;

		/* try to update pi-blocking */
		if (is_waitqueued(tracked)) {
			tsk_rt(tracked)->last_updated = now;
		}
	}
	entry->tracked = tracked;

#ifdef WANT_ALL_SCHED_EVENTS
	if (tracked)
		TRACE_TASK(tracked, "tracked to %d.\n", entry->cpu);
	else
		TRACE("NULL tracked to %d.\n", entry->cpu);
#endif
	update_cpu_position2(entry);
}

/* unlink - Make sure a task is not linked any longer to an entry
 *          where it was linked before. Must hold gsnedf_lock.
 */
static noinline void unlink(struct task_struct* t)
{
	cpu_entry_t *entry;

	if (t->rt_param.linked_on != NO_CPU) {
		/* unlink */
		entry = &per_cpu(gsnedf2_cpu_entries, t->rt_param.linked_on);
		t->rt_param.linked_on = NO_CPU;
		link_task_to_cpu(NULL, entry);
	} else if (is_queued(t)) {
		/* This is an interesting situation: t is scheduled,
		 * but was just recently unlinked.  It cannot be
		 * linked anywhere else (because then it would have
		 * been relinked to this CPU), thus it must be in some
		 * queue. We must remove it from the list in this
		 * case.
		 */
		remove(&gsnedf, t);
	}
}

// Similar to unlinking, untracking will remove a task from
// the cpu in terms of tracking the top m.
static noinline void untrack(struct task_struct* t)
{
	cpu_entry_t *entry;

	if (t->rt_param.tracked_on != NO_CPU) {
		/* unlink */
		entry = &per_cpu(gsnedf2_cpu_entries, t->rt_param.tracked_on);
		t->rt_param.tracked_on = NO_CPU;
		track_task_to_cpu(NULL, entry);

		try_update_pi_blocking(t, 0);
	} else if (is_queued2(t)) {
		remove2(&gsnedf, t);
	}
}

static void update_queue_position(struct task_struct *t);
static void update_queue_position2(struct task_struct *t);

// This is the hrtimer callback that will run when a task
// reaches zero laxity. This function will boost the task's priority.
static enum hrtimer_restart on_zero_laxity(struct hrtimer *timer)
{
	unsigned long flags;
	struct task_struct* t;

	TS_SCHED_TIMER_START
	raw_spin_lock_irqsave(&gsnedf_lock, flags);

	// Grab the task of this timer
	t = container_of(container_of(timer, struct rt_param, zl_timer),
			struct task_struct,
			rt_param);

	// Set the laxity flag, and update queue positions. We update
	// all queues because this will very likely affect actual scheduling.
	set_zerolaxity(t);
	update_queue_position(t);
	update_queue_position2(t);

	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);
	TS_SCHED_TIMER_END

	return HRTIMER_NORESTART;
}

// For EDZL, if we are taking a ready job, make sure to cancel
// the zero laxity timer associated with it. Other than that, this is
// just the take ready function which will retrieve a ready task.
static inline struct task_struct* __edzl_take_ready(rt_domain_t* rt)
{
	struct task_struct* t = __take_ready(rt);

	if (t) {
		if (get_zerolaxity(t) == 0) {
			if (hrtimer_active(&tsk_rt(t)->zl_timer)) {
				hrtimer_try_to_cancel(&tsk_rt(t)->zl_timer);
			}
		}
	}
	return t;
}

// Adds a ready task to this scheduler. This function will also set up
// an hr timer to ensure the zero_laxity event can be captured.
static inline void __edzl_add_ready(rt_domain_t* rt, struct task_struct *new)
{
	__add_ready(rt, new);

	if (get_zerolaxity(new) == 0) {
		lt_t when_to_fire;

		when_to_fire = get_deadline(new) - budget_remaining(new);

		hrtimer_start_range_ns(&tsk_rt(new)->zl_timer,
				ns_to_ktime(when_to_fire),
				0,
				HRTIMER_MODE_ABS_PINNED);
	}
}

static void check_for_preemptions(void);
static struct task_struct* find_hp_waiter(struct njlp_semaphore *sem);

// Updating a task's queue position will likely happen when there
// is a priority change. The overview of this function is to check
// and update all appropriate queues related to scheduling appropriately.
static void update_queue_position(struct task_struct *t)
{
	int check_preempt = 0;
	struct njlp_semaphore* sem;
	unsigned long flags;

	if (is_waitqueued(t)) {
		// If we are wait queued, make sure we update our
		// position in this wait queue

		// Grab the associated lock
		sem = njlp_from_lock(tsk_rt(t)->sem);
		BUG_ON(!sem);

		spin_lock(&sem->qlock);
		if (is_waitqueued(t)) {
			BUG_ON(!bheap_node_in_heap(tsk_rt(t)->waitq_heap_node2));

			// Update our position in the wait queue
			bheap_decrease(edzl_ready_order, tsk_rt(t)->waitq_heap_node2);

			// Is priority inheritance needed?
			sem->hp_waiter = find_hp_waiter(sem);
			if (sem->hp_waiter == t)
				tsk_rt(sem->owner)->inh_task = t;
		}
		spin_unlock(&sem->qlock);
	}

	// Is this task on the CPU?
	if (tsk_rt(t)->linked_on != NO_CPU) {
		// Re-enqueue it to update the queue position
		bheap_delete(cpu_lower_prio, &gsnedf_cpu_heap,
				gsnedf2_cpus[tsk_rt(t)->linked_on]->hn);
		bheap_insert(cpu_lower_prio, &gsnedf_cpu_heap,
				gsnedf2_cpus[tsk_rt(t)->linked_on]->hn);
	} else {
		// If we aren't on the CPU, then..
		raw_spin_lock(&gsnedf.release_lock);
		if (is_queued(t)) {
			// Update our ready queue position
			check_preempt = !bheap_decrease(edzl_ready_order,
					tsk_rt(t)->heap_node);
		}
		raw_spin_unlock(&gsnedf.release_lock);

		// If we're at the top of the ready queue, then
		// we should check for preemption opportunities
		if (check_preempt) {
			bheap_uncache_min(edzl_ready_order,
					&gsnedf.ready_queue);
			check_for_preemptions();
		}
	}
}

static void check_for_prio_changes(void);

// This is similar to the update_queue_position function but less
// needs to be done here because we don't need to actually schedule
// on the cpu, but only where tasks are tracked
static void update_queue_position2(struct task_struct *t)
{
	int check_preempt = 0;

	if (tsk_rt(t)->tracked_on != NO_CPU) {
		// If we are tracked on a CPU, then update that as
		// our position may have changed
		bheap_delete(cpu_lower_base_prio, &gsnedf_cpu_heap,
				gsnedf2_cpus[tsk_rt(t)->tracked_on]->hn);
		bheap_insert(cpu_lower_base_prio, &gsnedf_cpu_heap,
				gsnedf2_cpus[tsk_rt(t)->tracked_on]->hn);
	} else {
		// If we aren't tracked on the CPU..
		raw_spin_lock(&gsnedf.release_lock);
		if (is_queued2(t)) {
			// Update our position in the secondary ready queue
			check_preempt = !bheap_decrease(edzl_pending_order,
					tsk_rt(t)->heap_node);
		}
		raw_spin_unlock(&gsnedf.release_lock);

		// If we hit the top of the ready queue, then it is likely
		// that we need to check for priority changes where this task
		// might need to be tracked on a CPU instead
		if (check_preempt) {
			bheap_uncache_min(edzl_pending_order,
					&gsnedf.pending_queue);
			check_for_prio_changes();
		}
	}
}

/* preempt - force a CPU to reschedule
 */
static void preempt(cpu_entry_t *entry)
{
	preempt_if_preemptable(entry->scheduled, entry->cpu);
}

/* requeue - Put an unlinked task into gsn-edf domain.
 *           Caller must hold gsnedf_lock.
 */
static noinline void requeue(struct task_struct* task)
{
	BUG_ON(!task);
	/* sanity check before insertion */
	BUG_ON(is_queued(task));

	if (is_early_releasing(task) || is_released(task, litmus_clock()))
		__edzl_add_ready(&gsnedf, task);
	else {
		/* it has got to wait */
		untrack(task);
		add_release(&gsnedf, task);
	}
}

// This function is similar to requeue except operating on the
// secondary data structures.
static noinline void requeue2(struct task_struct* task)
{
	BUG_ON(!task);

	if (is_early_releasing(task) || is_released(task, litmus_clock())) {
		/* sanity check before insertion */
		BUG_ON(is_queued2(task));
		if (!is_queued2(task))
			__add_pending(&gsnedf, task);
	}
	// requeue should take care of adding things to the release Q for us
}

// The EDZL-NJLP does not incorporate affinity masks
#ifdef CONFIG_SCHED_CPU_AFFINITY
static cpu_entry_t* gsnedf_get_nearest_available_cpu(cpu_entry_t *start)
{
	cpu_entry_t *affinity;

	get_nearest_available_cpu(affinity, start, gsnedf2_cpu_entries,
#ifdef CONFIG_RELEASE_MASTER
			gsnedf.release_master,
#else
			NO_CPU,
#endif
			cpu_online_mask);

	return(affinity);
}
#endif

/* check for any necessary preemptions */
static void check_for_preemptions(void)
{
	struct task_struct *task;
	cpu_entry_t *last;


#ifdef CONFIG_PREFER_LOCAL_LINKING
	cpu_entry_t *local;

	/* Before linking to other CPUs, check first whether the local CPU is
	 * idle. */
	local = this_cpu_ptr(&gsnedf2_cpu_entries);
	task  = __peek_ready(&gsnedf);

	if (task && !local->linked
#ifdef CONFIG_RELEASE_MASTER
	    && likely(local->cpu != gsnedf.release_master)
#endif
		) {
		// Local cpu is idle, so link to the local CPU
		task = __edzl_take_ready(&gsnedf);
		TRACE_TASK(task, "linking to local CPU %d to avoid IPI\n", local->cpu);
		link_task_to_cpu(task, local);
		preempt(local);
	}
#endif

	// This for loop is simple, find the lowest priority job on the CPU
	// and check to see if it is a lower priority than the highest priority
	// job that is ready. If so, preempt, swap the two tasks, then repeat.
	for (last = lowest_prio_cpu();
	     edzl_preemption_needed(&gsnedf, last->linked);
	     last = lowest_prio_cpu()) {
		/* preemption necessary */
		task = __edzl_take_ready(&gsnedf);
		TRACE("check_for_preemptions: attempting to link task %d to %d\n",
		      task->pid, last->cpu);

#ifdef CONFIG_SCHED_CPU_AFFINITY
		{
			cpu_entry_t *affinity =
					gsnedf_get_nearest_available_cpu(
						&per_cpu(gsnedf2_cpu_entries, task_cpu(task)));
			if (affinity)
				last = affinity;
			else if (requeue_preempted_job(last->linked))
				requeue(last->linked);
		}
#else
		// We need to requeue the job that was preempted
		if (requeue_preempted_job(last->linked))
			requeue(last->linked);
#endif

		// Add the new task to the cpu
		link_task_to_cpu(task, last);
		preempt(last);
	}
}

// This is similar to the function above, but for the NJLP's doubled
// scheduling logic. It may look smaller, but that is because we do not
// incorporate affinity masks in this work.
static void check_for_prio_changes(void)
{
	struct task_struct *task;
	cpu_entry_t *last;


#ifdef CONFIG_PREFER_LOCAL_LINKING
	cpu_entry_t *local;

	/* Before linking to other CPUs, check first whether the local CPU is
	 * idle. */
	local = this_cpu_ptr(&gsnedf2_cpu_entries);
	task  = __peek_pending(&gsnedf);

	if (task && !local->tracked
#ifdef CONFIG_RELEASE_MASTER
	    && likely(local->cpu != gsnedf.release_master)
#endif
		) {
		// Local CPU is idle, so link to the local CPU
		task = __take_pending(&gsnedf);
		TRACE_TASK(task, "tracking to local CPU %d to mimic local linking\n", local->cpu);
		track_task_to_cpu(task, local);
	}
#endif

	// This for loop is simple, find the lowest priority job on the CPU
	// and check to see if it is a lower priority than the highest priority
	// job that is ready. If so, preempt, swap the two tasks, then repeat.
	for (last = lowest_base_prio_cpu();
	     edzl_preemption_needed2(&gsnedf, last->tracked);
	     last = lowest_base_prio_cpu()) {
		/* preemption necessary */
		task = __take_pending(&gsnedf);
		TRACE("check_for_prio_changes: attempting to track task %d to %d\n",
		      task->pid, last->cpu);

		if (requeue_preempted_job(last->tracked))
			requeue2(last->tracked);

		track_task_to_cpu(task, last);
	}
}

/* gsnedf_job_arrival: task is either resumed or released */
static noinline void gsnedf_job_arrival(struct task_struct* task)
{
	BUG_ON(!task);

	// Update laxity.
	if (laxity_remaining(task))
		clear_zerolaxity(task);
	else
		set_zerolaxity(task);

	// will enqueue the task so that we can schedule it
	requeue(task);
	// Possible that this task will need to be scheduled immediately
	check_for_preemptions();
}

/* pending jobs prio ordering changed due to task release */
static noinline void gsnedf_prio_change(struct task_struct* task)
{
	BUG_ON(!task);

	// All we need to do is requeue the task with the priority change
	requeue2(task);
	// possible we will need to track on to a CPU if we're in the top m
	check_for_prio_changes();
}

// This will merge the tasks into the ready queue
static void gsnedf_release_jobs(rt_domain_t* rt, struct bheap* tasks)
{
	unsigned long flags;

	raw_spin_lock_irqsave(&gsnedf_lock, flags);

	// Merge incoming tasks into the ready queue
	__merge_ready(rt, tasks);
	// Possible some of these tasks may need to be linked to a CPU
	check_for_preemptions();

	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);
}

// doubled scheduling logic version of the above function.
static void gsnedf_release_jobs2(rt_domain_t* rt, struct bheap* tasks)
{
	unsigned long flags;

	raw_spin_lock_irqsave(&gsnedf_lock, flags);

	__merge_pending(rt, tasks);
	// It is possible that some of these  tasks may be in the top
	// m priorities, so check for prio change to track them.
	check_for_prio_changes();

	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);
}

/* caller holds gsnedf_lock */
static noinline void curr_job_completion(int forced)
{
	struct task_struct *t = current;
	BUG_ON(!t);

	// This is just additional trace outputs
	sched_trace_task_completion(t, forced);

	TRACE_TASK(t, "job_completion(forced=%d).\n", forced);

	/* set flags */
	tsk_rt(t)->completed = 0;
	/* prepare for next period */
	prepare_for_next_period(t);
	if (is_early_releasing(t) || is_released(t, litmus_clock()))
		sched_trace_task_release(t);
	/* unlink */
	unlink(t);
	untrack(t);
	/* requeue
	 * But don't requeue a blocking task. */
	if (is_current_running()) {
		gsnedf_job_arrival(t);
		gsnedf_prio_change(t);
	}
}

/* Getting schedule() right is a bit tricky. schedule() may not make any
 * assumptions on the state of the current task since it may be called for a
 * number of reasons. The reasons include a scheduler_tick() determined that it
 * was necessary, because sys_exit_np() was called, because some Linux
 * subsystem determined so, or even (in the worst case) because there is a bug
 * hidden somewhere. Thus, we must take extreme care to determine what the
 * current state is.
 *
 * The CPU could currently be scheduling a task (or not), be linked (or not).
 *
 * The following assertions for the scheduled task could hold:
 *
 *      - !is_running(scheduled)        // the job blocks
 *	- scheduled->timeslice == 0	// the job completed (forcefully)
 *	- is_completed()		// the job completed (by syscall)
 * 	- linked != scheduled		// we need to reschedule (for any reason)
 * 	- is_np(scheduled)		// rescheduling must be delayed,
 *					   sys_exit_np must be requested
 *
 * Any of these can occur together.
 *
 * We need to return the task that we do want scheduled though, and NULL if
 * we're okay with being idle.
 */
static struct task_struct* gsnedf_schedule(struct task_struct * prev)
{
	cpu_entry_t* entry = this_cpu_ptr(&gsnedf2_cpu_entries);
	int out_of_time, sleep, preempt, np, exists, blocks;
	struct task_struct* next = NULL;

#ifdef CONFIG_RELEASE_MASTER
	/* Bail out early if we are the release master.
	 * The release master never schedules any real-time tasks.
	 */
	if (unlikely(gsnedf.release_master == entry->cpu)) {
		sched_state_task_picked();
		return NULL;
	}
#endif

	raw_spin_lock(&gsnedf_lock);

	/* sanity checking */
	BUG_ON(entry->scheduled && entry->scheduled != prev);
	BUG_ON(entry->scheduled && !is_realtime(prev));
	BUG_ON(is_realtime(prev) && !entry->scheduled);

	/* (0) Determine state */
	exists      = entry->scheduled != NULL;
	blocks      = exists && !is_current_running();
	out_of_time = exists && budget_enforced(entry->scheduled)
		&& budget_exhausted(entry->scheduled);
	np 	    = exists && is_np(entry->scheduled);
	sleep	    = exists && is_completed(entry->scheduled);
	preempt     = entry->scheduled != entry->linked;

#ifdef WANT_ALL_SCHED_EVENTS
	TRACE_TASK(prev, "invoked gsnedf_schedule.\n");
#endif

	if (exists)
		TRACE_TASK(prev,
			   "blocks:%d out_of_time:%d np:%d sleep:%d preempt:%d "
			   "state:%d sig:%d\n",
			   blocks, out_of_time, np, sleep, preempt,
			   prev->state, signal_pending(prev));
	if (entry->linked && preempt)
		TRACE_TASK(prev, "will be preempted by %s/%d\n",
			   entry->linked->comm, entry->linked->pid);


	/* If a task blocks we have no choice but to reschedule.
	 */
	if (blocks)
		unlink(entry->scheduled);

	/* Request a sys_exit_np() call if we would like to preempt but cannot.
	 * We need to make sure to update the link structure anyway in case
	 * that we are still linked. Multiple calls to request_exit_np() don't
	 * hurt.
	 */
	if (np && (out_of_time || preempt || sleep)) {
		untrack(entry->scheduled);
		unlink(entry->scheduled);
		request_exit_np(entry->scheduled);
	}

	/* Any task that is preemptable and either exhausts its execution
	 * budget or wants to sleep completes. We may have to reschedule after
	 * this. Don't do a job completion if we block (can't have timers running
	 * for blocked jobs).
	 */
	if (!np && (out_of_time || sleep))
		curr_job_completion(!sleep);

	/* Link pending task if we became unlinked.
	 */
	if (!entry->linked)
		link_task_to_cpu(__edzl_take_ready(&gsnedf), entry);
	if (!entry->tracked)
		track_task_to_cpu(__take_pending(&gsnedf), entry);

	/* The final scheduling decision. Do we need to switch for some reason?
	 * If linked is different from scheduled, then select linked as next.
	 */
	if ((!np || blocks) &&
	    entry->linked != entry->scheduled) {
		/* Schedule a linked job? */
		if (entry->linked) {
			entry->linked->rt_param.scheduled_on = entry->cpu;
			next = entry->linked;
			TRACE_TASK(next, "scheduled_on = P%d\n", smp_processor_id());
		}
		if (entry->scheduled) {
			/* not gonna be scheduled soon */
			entry->scheduled->rt_param.scheduled_on = NO_CPU;
			TRACE_TASK(entry->scheduled, "scheduled_on = NO_CPU\n");
		}
	} else
		/* Only override Linux scheduler if we have a real-time task
		 * scheduled that needs to continue.
		 */
		if (exists)
			next = prev;

	sched_state_task_picked();

	raw_spin_unlock(&gsnedf_lock);

#ifdef WANT_ALL_SCHED_EVENTS
	TRACE("gsnedf_lock released, next=0x%p\n", next);

	if (next)
		TRACE_TASK(next, "scheduled at %llu\n", litmus_clock());
	else if (exists && !next)
		TRACE("becomes idle at %llu.\n", litmus_clock());
#endif


	return next;
}


/* _finish_switch - we just finished the switch away from prev
 */
static void gsnedf_finish_switch(struct task_struct *prev)
{
	cpu_entry_t* 	entry = this_cpu_ptr(&gsnedf2_cpu_entries);

	entry->scheduled = is_realtime(current) ? current : NULL;
#ifdef WANT_ALL_SCHED_EVENTS
	TRACE_TASK(prev, "switched away from\n");
#endif
}


/*	Prepare a task for running in RT mode
 */
static void gsnedf_task_new(struct task_struct * t, int on_rq, int is_scheduled)
{
	unsigned long 		flags;
	cpu_entry_t* 		entry;

	TRACE("gsn edf: task new %d\n", t->pid);

	raw_spin_lock_irqsave(&gsnedf_lock, flags);

	// This hrtimer will help find our zero laxity point if we get boosted.
	hrtimer_init(&t->rt_param.zl_timer, CLOCK_MONOTONIC, HRTIMER_MODE_ABS);
	tsk_rt(t)->zl_timer.function = on_zero_laxity;

	/* setup job params */
	release_at(t, litmus_clock());

	if (is_scheduled) {
		entry = &per_cpu(gsnedf2_cpu_entries, task_cpu(t));
		BUG_ON(entry->scheduled);

#ifdef CONFIG_RELEASE_MASTER
		if (entry->cpu != gsnedf.release_master) {
#endif
			// This cpu schedules t
			entry->scheduled = t;
			tsk_rt(t)->scheduled_on = task_cpu(t);
#ifdef CONFIG_RELEASE_MASTER
		} else {
			/* do not schedule on release master */
			preempt(entry); /* force resched */
			tsk_rt(t)->scheduled_on = NO_CPU;
		}
#endif
	} else {
		t->rt_param.scheduled_on = NO_CPU;
	}

	// New task isn't linked/tracked (yet)
	t->rt_param.linked_on          = NO_CPU;
	t->rt_param.tracked_on		   = NO_CPU;

	if (on_rq || is_scheduled) {
		// If we're ready or scheduled, then make sure this
		// new task arrival is properly linked and tracked
		// according to its priority
		gsnedf_job_arrival(t);
		gsnedf_prio_change(t);
	}
	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);
}

// When a task wakes up
static void gsnedf_task_wake_up(struct task_struct *task)
{
	unsigned long flags;
	lt_t now;

	TRACE_TASK(task, "wake_up at %llu\n", litmus_clock());

	raw_spin_lock_irqsave(&gsnedf_lock, flags);
	now = litmus_clock();
	if (is_sporadic(task) && is_tardy(task, now)) {
		inferred_sporadic_job_release_at(task, now);
	}
	gsnedf_job_arrival(task);
	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);
}

// When a task blocks, we unlink it so it isn't on a cpu
static void gsnedf_task_block(struct task_struct *t)
{
	unsigned long flags;

	TRACE_TASK(t, "block at %llu\n", litmus_clock());

	/* unlink if necessary */
	raw_spin_lock_irqsave(&gsnedf_lock, flags);
	unlink(t);
	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);

	BUG_ON(!is_realtime(t));
}

// Remove the task from the romain
static void gsnedf_task_exit(struct task_struct * t)
{
	unsigned long flags;

	/* unlink if necessary */
	raw_spin_lock_irqsave(&gsnedf_lock, flags);
	unlink(t);
	untrack(t);
	if (tsk_rt(t)->scheduled_on != NO_CPU) {
		// Remove ourselves from the CPU
		gsnedf2_cpus[tsk_rt(t)->scheduled_on]->scheduled = NULL;
		tsk_rt(t)->scheduled_on = NO_CPU;
	}
	if (tsk_rt(t)->tracked_on != NO_CPU) {
		// Remove ourselves from the top m priorities
		gsnedf2_cpus[tsk_rt(t)->tracked_on]->tracked = NULL;
		tsk_rt(t)->tracked_on = NO_CPU;
	}

	// cancel the zero laxity timer
	if (hrtimer_active(&tsk_rt(t)->zl_timer)) {
		hrtimer_cancel(&tsk_rt(t)->zl_timer);
	}
	raw_spin_unlock_irqrestore(&gsnedf_lock, flags);

	BUG_ON(!is_realtime(t));
        TRACE_TASK(t, "RIP\n");
}

// Just admit all of them
static long gsnedf_admit_task(struct task_struct* tsk)
{
	return 0;
}

#ifdef CONFIG_LITMUS_LOCKING

#include <litmus/fdso.h>

/* called with IRQs off */
// Priority inheritance is used to ensure the system doesn't stall. If a higher-priority job
// is waiting on a resource held by a lower priority job, the lower priority job inherits the
// higher priority so that the higher priority job isn't waiting on a job that doesn't have
// sufficient priority. Of course, the higher priority might also not be sufficient priority
// to be scheduled, but that blocking time is attributed to higher-priority work rather than
// the priority inversion blocking time.
static void set_priority_inheritance(struct task_struct* t, struct task_struct* prio_inh)
{
	int linked_on;
	int check_preempt = 0;

	raw_spin_lock(&gsnedf_lock);

	TRACE_TASK(t, "inherits priority from %s/%d\n", prio_inh->comm, prio_inh->pid);
	tsk_rt(t)->inh_task = prio_inh;

	// which cpu are we linked on?
	linked_on  = tsk_rt(t)->linked_on;

	/* If it is scheduled, then we need to reorder the CPU heap. */
	if (linked_on != NO_CPU) {
		TRACE_TASK(t, "%s: linked  on %d\n",
			   __FUNCTION__, linked_on);
		/* Holder is scheduled; need to re-order CPUs.
		 * We can't use heap_decrease() here since
		 * the cpu_heap is ordered in reverse direction, so
		 * it is actually an increase. */
		bheap_delete(cpu_lower_prio, &gsnedf_cpu_heap,
			    gsnedf2_cpus[linked_on]->hn);
		bheap_insert(cpu_lower_prio, &gsnedf_cpu_heap,
			    gsnedf2_cpus[linked_on]->hn);
	} else {
		/* holder may be queued: first stop queue changes */
		raw_spin_lock(&gsnedf.release_lock);
		if (is_queued(t)) {
			TRACE_TASK(t, "%s: is queued\n",
				   __FUNCTION__);
			/* We need to update the position of holder in some
			 * heap. Note that this could be a release heap if we
			 * budget enforcement is used and this job overran. */
			check_preempt =
				!bheap_decrease(edzl_ready_order,
					       tsk_rt(t)->heap_node);
		} else {
			/* Nothing to do: if it is not queued and not linked
			 * then it is either sleeping or currently being moved
			 * by other code (e.g., a timer interrupt handler) that
			 * will use the correct priority when enqueuing the
			 * task. */
			TRACE_TASK(t, "%s: is NOT queued => Done.\n",
				   __FUNCTION__);
		}
		raw_spin_unlock(&gsnedf.release_lock);

		/* If holder was enqueued in a release heap, then the following
		 * preemption check is pointless, but we can't easily detect
		 * that case. If you want to fix this, then consider that
		 * simply adding a state flag requires O(n) time to update when
		 * releasing n tasks, which conflicts with the goal to have
		 * O(log n) merges. */
		if (check_preempt) {
			/* heap_decrease() hit the top level of the heap: make
			 * sure preemption checks get the right task, not the
			 * potentially stale cache. */
			bheap_uncache_min(edzl_ready_order,
					 &gsnedf.ready_queue);
			check_for_preemptions();
		}
	}

	raw_spin_unlock(&gsnedf_lock);
}

/* called with IRQs off */
static void clear_priority_inheritance(struct task_struct* t)
{
	raw_spin_lock(&gsnedf_lock);

	/* A job only stops inheriting a priority when it releases a
	 * resource. Thus we can make the following assumption.*/
	BUG_ON(tsk_rt(t)->scheduled_on == NO_CPU);

	TRACE_TASK(t, "priority restored\n");
	tsk_rt(t)->inh_task = NULL;

	/* Check if rescheduling is necessary. We can't use heap_decrease()
	 * since the priority was effectively lowered. */
	unlink(t);
	gsnedf_job_arrival(t);

	raw_spin_unlock(&gsnedf_lock);
}


/* ******************** FMLP/NJLP support ********************** */

static inline void add_waitqueue(struct njlp_semaphore *sem, struct task_struct* new)
{
	spin_lock(&sem->qlock);

	BUG_ON(bheap_node_in_heap(tsk_rt(new)->waitq_heap_node));
	BUG_ON(tsk_rt(new)->sem != &sem->litmus_lock);

	// Doubled scheduling logic. NJLP priority order is by piblocking
	// and EDZL priority order is by EDF and zero laxity.
	bheap_insert(njlp_priority_order, &sem->waitq, tsk_rt(new)->waitq_heap_node);
	bheap_insert(edzl_ready_order, &sem->waitq2, tsk_rt(new)->waitq_heap_node2);

	spin_unlock(&sem->qlock);
}

// Grab the next task that is in this semaphore's wait queue
static inline struct task_struct* take_waitqueue(struct njlp_semaphore *sem)
{
	struct task_struct* t = NULL;
	struct bheap_node* hn;

	spin_lock(&sem->qlock);

	hn = bheap_take(njlp_priority_order, &sem->waitq);
	if (hn) {
		// Get the task
		t = bheap2task(hn);
		BUG_ON(is_waitqueued(t));
		// And remove it from the wait queue
		bheap_delete(edzl_ready_order, &sem->waitq2, tsk_rt(t)->waitq_heap_node2);
	}

	spin_unlock(&sem->qlock);

	return t;
}

/* caller is responsible for locking */
static struct task_struct* find_hp_waiter(struct njlp_semaphore *sem)
{
	struct bheap_node* hn = bheap_peek(edzl_ready_order, &sem->waitq2);
	if (hn)
		return bheap2task(hn);
	else
		return NULL;
}

// This function is called when locking on a lock arbitrated by the NJLP.
int gsnedf_njlp_lock(struct litmus_lock* l)
{
	struct task_struct* t = current;
	struct njlp_semaphore *sem = njlp_from_lock(l);
	unsigned long flags;

	if (!is_realtime(t))
		return -EPERM;

	/* prevent nested lock acquisition --- not supported by FMLP */
	if (tsk_rt(t)->num_locks_held)
		return -EBUSY;

	// trace locking overheads
	TS_LOCK_START

	spin_lock_irqsave(&sem->waitlock, flags);

	if (sem->owner) {
		/* resource is not free => must suspend and wait */

		tsk_rt(t)->pi_blocked = 0;
		tsk_rt(t)->last_updated = litmus_clock();
		tsk_rt(t)->sem = &sem->litmus_lock;

		/* FIXME: interruptible would be nice some day */
		set_current_state(TASK_UNINTERRUPTIBLE);

		add_waitqueue(sem, t);

		/* check if we need to activate priority inheritance */
		if (edzl_higher_prio(t, sem->hp_waiter)) {
			sem->hp_waiter = t;
			if (edzl_higher_prio(t, sem->owner))
				set_priority_inheritance(sem->owner, sem->hp_waiter);
		}

		TS_LOCK_END
		TS_LOCK_SUSPEND;

		/* release lock before sleeping */
		spin_unlock_irqrestore(&sem->waitlock, flags);

		/* We depend on the FIFO order.  Thus, we don't need to recheck
		 * when we wake up; we are guaranteed to have the lock since
		 * there is only one wake up per release.
		 */

		schedule();

		TS_LOCK_RESUME;

		/* Since we hold the lock, no other task will change
		 * ->owner. We can thus check it without acquiring the spin
		 * lock. */
		BUG_ON(sem->owner != t);
	} else {
		/* it's ours now */
		sem->owner = t;
		TS_LOCK_END

		spin_unlock_irqrestore(&sem->waitlock, flags);
	}

	tsk_rt(t)->num_locks_held++;

	return 0;
}

// This function is called when unlocking with the NJLP
int gsnedf_njlp_unlock(struct litmus_lock* l)
{
	struct task_struct *t = current, *next;
	struct njlp_semaphore *sem = njlp_from_lock(l);
	unsigned long flags;
	int err = 0;

	int cpu;
	cpu_entry_t *entry;

	// trace the unlock overheads
	TS_UNLOCK_START

	spin_lock_irqsave(&sem->waitlock, flags);

	if (sem->owner != t) {
		err = -EINVAL;
		goto out;
	}

	tsk_rt(t)->num_locks_held--;

	/* only do pi-blocking updates if there are suspended jobs */
	if (bheap_peek(njlp_priority_order, &sem->waitq)) {
		/* loop through top m prio pending jobs and update pi-blocking*/
		raw_spin_lock(&gsnedf_lock);
		for_each_online_cpu(cpu) {
			entry = &per_cpu(gsnedf2_cpu_entries, cpu);
			try_update_pi_blocking(entry->tracked, 1);
		}
		raw_spin_unlock(&gsnedf_lock);
	}

	/* check if there are jobs waiting for this resource */
	next = take_waitqueue(sem);

	if (next) {
		BUG_ON(tsk_rt(next)->sem != l);
		tsk_rt(next)->sem = NULL;
		tsk_rt(next)->pi_blocked = 0;
		BUG_ON(is_waitqueued(next));

		/* next becomes the resouce holder */
		sem->owner = next;
		TRACE_CUR("lock ownership passed to %s/%d\n", next->comm, next->pid);

		/* determine new hp_waiter if necessary */
		if (next == sem->hp_waiter) {
			TRACE_TASK(next, "was highest-prio waiter\n");
			/* next has the highest priority --- it doesn't need to
			 * inherit.  However, we need to make sure that the
			 * next-highest priority in the queue is reflected in
			 * hp_waiter. */
			sem->hp_waiter = find_hp_waiter(sem);
			if (sem->hp_waiter)
				TRACE_TASK(sem->hp_waiter, "is new highest-prio waiter\n");
			else
				TRACE("no further waiters\n");
		} else {
			/* Well, if next is not the highest-priority waiter,
			 * then it ought to inherit the highest-priority
			 * waiter's priority. */
			set_priority_inheritance(next, sem->hp_waiter);
		}

		/* wake up next */
		wake_up_process(next);
	} else
		/* becomes available */
		sem->owner = NULL;

	/* we lose the benefit of priority inheritance (if any) */
	if (tsk_rt(t)->inh_task)
		clear_priority_inheritance(t);

out:
	spin_unlock_irqrestore(&sem->waitlock, flags);
	TS_UNLOCK_END

	return err;
}

// Function will unlock the lock if we are the owner
int gsnedf_njlp_close(struct litmus_lock* l)
{
	struct task_struct *t = current;
	struct njlp_semaphore *sem = njlp_from_lock(l);
	unsigned long flags;

	int owner;

	spin_lock_irqsave(&sem->waitlock, flags);

	owner = sem->owner == t;

	spin_unlock_irqrestore(&sem->waitlock, flags);

	if (owner)
		gsnedf_njlp_unlock(l);

	return 0;
}

// Free the lock data structure
void gsnedf_njlp_free(struct litmus_lock* lock)
{
	kfree(njlp_from_lock(lock));
}

// Litmus requires us to specify how lock operations are done
// using a struct of function pointers.
static struct litmus_lock_ops gsnedf_njlp_lock_ops = {
	.close  = gsnedf_njlp_close,
	.lock   = gsnedf_njlp_lock,
	.unlock = gsnedf_njlp_unlock,
	.deallocate = gsnedf_njlp_free,
};

// Initialize the data structures associated with an NJLP lock
static struct litmus_lock* gsnedf_new_njlp(void)
{
	struct njlp_semaphore* sem;

	sem = kmalloc(sizeof(*sem), GFP_KERNEL);
	if (!sem)
		return NULL;

	// This just initializes the struct
	sem->owner   = NULL;
	sem->hp_waiter = NULL;
	bheap_init(&sem->waitq);
	bheap_init(&sem->waitq2);
	spin_lock_init(&sem->waitlock);
	spin_lock_init(&sem->qlock);
	sem->litmus_lock.ops = &gsnedf_njlp_lock_ops;

	return &sem->litmus_lock;
}

/* **** lock constructor **** */
static long gsnedf_allocate_lock(struct litmus_lock **lock, int type,
				 void* __user unused)
{
	int err = -ENXIO;

	/* EDZL-NJLP currently only supports the NJLP for global resources. */
	switch (type) {

	case NJLP_SEM:
		/* non-JLFP Locking Protocol */
		*lock = gsnedf_new_njlp();
		if (*lock)
			err = 0;
		else
			err = -ENOMEM;
		break;

	};

	return err;
}

#endif

static struct domain_proc_info gsnedf_domain_proc_info;
static long gsnedf_get_domain_proc_info(struct domain_proc_info **ret)
{
	*ret = &gsnedf_domain_proc_info;
	return 0;
}

// This function initializes everything associated with scheduling on
// EDZL-NJLP, including cpu queues and initializing the rt_domain.
static void gsnedf_setup_domain_proc(void)
{
	int i, cpu;
	int release_master =
#ifdef CONFIG_RELEASE_MASTER
			atomic_read(&release_master_cpu);
#else
		NO_CPU;
#endif
	int num_rt_cpus = num_online_cpus() - (release_master != NO_CPU);
	struct cd_mapping *map;

	// init the domain
	memset(&gsnedf_domain_proc_info, 0, sizeof(gsnedf_domain_proc_info));
	init_domain_proc_info(&gsnedf_domain_proc_info, num_rt_cpus, 1);
	gsnedf_domain_proc_info.num_cpus = num_rt_cpus;
	gsnedf_domain_proc_info.num_domains = 1;

	// init the CPUs
	gsnedf_domain_proc_info.domain_to_cpus[0].id = 0;
	for (cpu = 0, i = 0; cpu < num_online_cpus(); ++cpu) {
		if (cpu == release_master)
			continue;
		map = &gsnedf_domain_proc_info.cpu_to_domains[i];
		map->id = cpu;
		cpumask_set_cpu(0, map->mask);
		++i;

		/* add cpu to the domain */
		cpumask_set_cpu(cpu,
			gsnedf_domain_proc_info.domain_to_cpus[0].mask);
	}
}

// When the EDZL-NJLP plugin is activated, this gets called
static long gsnedf_activate_plugin(void)
{
	int cpu;
	cpu_entry_t *entry;

	bheap_init(&gsnedf_cpu_heap);
	bheap_init(&gsnedf_cpu_heap2);
#ifdef CONFIG_RELEASE_MASTER
	gsnedf.release_master = atomic_read(&release_master_cpu);
#endif

	// Each of the cpus has data structures associated with it.
	// The loop below simply initializes each cpu's data structures.
	
	for_each_online_cpu(cpu) {
		entry = &per_cpu(gsnedf2_cpu_entries, cpu);
		bheap_node_init(&entry->hn, entry);
		bheap_node_init(&entry->hn2, entry);
		entry->linked    = NULL;
		entry->scheduled = NULL;
		entry->tracked	 = NULL;
#ifdef CONFIG_RELEASE_MASTER
		if (cpu != gsnedf.release_master) {
#endif
			TRACE("GSN-EDF: Initializing CPU #%d.\n", cpu);
			update_cpu_position(entry);
			update_cpu_position2(entry);
#ifdef CONFIG_RELEASE_MASTER
		} else {
			TRACE("GSN-EDF: CPU %d is release master.\n", cpu);
		}
#endif
	}

	gsnedf_setup_domain_proc();

	return 0;
}

static long gsnedf_deactivate_plugin(void)
{
	destroy_domain_proc_info(&gsnedf_domain_proc_info);
	return 0;
}

/*	Plugin object	*/
// Litmus requires us to fill out function pointers and information
// related to our scheduler in the sched_plugin struct.
static struct sched_plugin gsn_edf_plugin __cacheline_aligned_in_smp = {
	.plugin_name		= "EDZL-NJLP",
	.finish_switch		= gsnedf_finish_switch,
	.task_new		= gsnedf_task_new,
	.complete_job		= complete_job,
	.task_exit		= gsnedf_task_exit,
	.schedule		= gsnedf_schedule,
	.task_wake_up		= gsnedf_task_wake_up,
	.task_block		= gsnedf_task_block,
	.admit_task		= gsnedf_admit_task,
	.activate_plugin	= gsnedf_activate_plugin,
	.deactivate_plugin	= gsnedf_deactivate_plugin,
	.get_domain_proc_info	= gsnedf_get_domain_proc_info,
#ifdef CONFIG_LITMUS_LOCKING
	.allocate_lock		= gsnedf_allocate_lock,
#endif
};

// This will be called when initializing the module, and happens very early
static int __init init_gsn_edf(void)
{
	int cpu;
	cpu_entry_t *entry;

	bheap_init(&gsnedf_cpu_heap);
	bheap_init(&gsnedf_cpu_heap2);
	/* initialize CPU state */
	for (cpu = 0; cpu < NR_CPUS; cpu++)  {
		entry = &per_cpu(gsnedf2_cpu_entries, cpu);
		gsnedf2_cpus[cpu] = entry;
		entry->cpu 	 = cpu;
		entry->hn        = &gsnedf_heap_node[cpu];
		entry->hn2		 = &gsnedf_heap_node2[cpu];
		bheap_node_init(&entry->hn, entry);
		bheap_node_init(&entry->hn2, entry);
	}
	// not having edf_domain_init2 is not a mistake
	edzl_domain_init2(&gsnedf, NULL, gsnedf_release_jobs, gsnedf_release_jobs2);
	return register_sched_plugin(&gsn_edf_plugin);
}

// Initialize this module
module_init(init_gsn_edf);
