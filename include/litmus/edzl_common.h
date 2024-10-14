/*
 * EDZL common data structures and utility functions shared by all EDZL
 * scheduler plugins
 */

/* CLEANUP: Add comments and make it less messy.
 *
 */

#ifndef __UNC_EDF_COMMON_H__
#define __UNC_EDF_COMMON_H__

#include <litmus/rt_domain.h>

void edzl_domain_init(rt_domain_t* rt, check_resched_needed_t resched,
		     release_jobs_t release);

void edzl_domain_init2(rt_domain_t* rt, check_resched_needed_t resched,
		     release_jobs_t release, release_jobs_t release2);

int edzl_higher_prio(struct task_struct* first,
		    struct task_struct* second);

int edzl_higher_base_prio(struct task_struct* first,
		    struct task_struct* second);

int edzl_ready_order(struct bheap_node* a, struct bheap_node* b);

int edzl_pending_order(struct bheap_node* a, struct bheap_node* b);

int edzl_preemption_needed(rt_domain_t* rt, struct task_struct *t);

int edzl_preemption_needed2(rt_domain_t* rt, struct task_struct *t);

#endif
