#ifndef _LITMUS_BUDGET_H_
#define _LITMUS_BUDGET_H_

/* Update the per-processor enforcement timer (arm/reproram/cancel) for
 * the next task. */
void update_enforcement_timer(struct task_struct* t);

inline static int budget_exhausted(struct task_struct* t)
{
	return get_exec_time(t) >= get_exec_cost(t);
}

inline static lt_t budget_remaining(struct task_struct* t)
{
	if (!budget_exhausted(t))
		return get_exec_cost(t) - get_exec_time(t);
	else
		/* avoid overflow */
		return 0;
}

#define get_zerolaxity(t)   (tsk_rt(t)->job_params.zero_laxity)
#define set_zerolaxity(t) (tsk_rt(t)->job_params.zero_laxity=1)
#define clear_zerolaxity(t) (tsk_rt(t)->job_params.zero_laxity=0)

inline static lt_t laxity_remaining(struct task_struct* t)
{
    lt_t now = litmus_clock();
	lt_t remaining = budget_remaining(t);
	lt_t deadline = get_deadline(t);

	if (lt_before(now + remaining, deadline))
		return (deadline - (now + remaining));
	else
		return 0;
}

#define budget_enforced(t) (tsk_rt(t)->task_params.budget_policy != NO_ENFORCEMENT)

#define budget_precisely_enforced(t) (tsk_rt(t)->task_params.budget_policy \
				      == PRECISE_ENFORCEMENT)

static inline int requeue_preempted_job(struct task_struct* t)
{
	/* Add task to ready queue only if not subject to budget enforcement or
	 * if the job has budget remaining. t may be NULL.
	 */
	return t && !is_completed(t) &&
		(!budget_exhausted(t) || !budget_enforced(t));
}

void litmus_current_budget(lt_t *used_so_far, lt_t *remaining);

#endif
