# Graph Dataset Pruning Formal-Evidence Reproduction

## Target claims and interpretation boundary

- Target claim: The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3). _(evidence: `/target_claims/0`)_
- Target claim: Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F). _(evidence: `/target_claims/1`)_

This report recomputes formal evidence from the released paper artifact. It does not present paper-reported experiments as reproduced measurements.

## Literal and repaired formulations

`paper_mwcp` and `paper_samplewise_literal` remain separate: the literal samplewise formulation double-counts symmetric pair interactions. `single_counted_pairwise` and `half_corrected_samplewise` are repaired comparisons, not paper implementations.
`appendix_inline_shift_literal` is the Appendix-inline quadratic shift. `modular_shift_candidate` is a distinct repaired modular candidate and never inherits the literal witness.
- MWCP witness value: 1/1 _(evidence: `/witnesses/2/comparison/paper_mwcp`)_
- Literal samplewise witness value: 2/1 _(evidence: `/witnesses/2/comparison/paper_samplewise_literal`)_

## Appendix shift and proof boundaries

The literal Appendix witness is the **1 then 3** marginal sequence _(evidence: `/witnesses/1/intermediate_values/marginal_empty` and `/witnesses/1/intermediate_values/marginal_y`)_; the exact rational values follow.
- First marginal: 1/1 _(evidence: `/witnesses/1/intermediate_values/marginal_empty`)_
- Second marginal: 3/1 _(evidence: `/witnesses/1/intermediate_values/marginal_y`)_
- Remainder cardinality: 2 _(evidence: `/witnesses/0/intermediate_values/remainder_cardinality`)_
- Claimed b-minus-t cardinality: 1 _(evidence: `/witnesses/0/intermediate_values/b_minus_t`)_
The cardinality record is independent of weights. It contradicts only the stated b-minus-t step; it is not labeled an Eq. counterexample or theorem counterexample.

## Greedy, optimum, and ratio evidence

The accepted aggregate retains exact ratio classifications and the independently recomputed greedy/optimum accounting. It does not retain terminal greedy and optimum values as displayable records, so this renderer does not invent them.
- Representative diagnostic: graph=n=3;vw=0/1,0/1,0/1;ew=-1/1,-1/1,0/1::variant=appendix_inline_shift_literal::alpha=1/1::eta=1/1::budget=2::selector=paper_eq7_score_greedy::result=canonical::path=v0,v1 _(evidence: `/out_of_premise_diagnostics/84/id`)_
- Ratio classification: defined_positive_optimum _(evidence: `/out_of_premise_diagnostics/84/ratio_classification/status`)_
- Exact ratio: 1/2 _(evidence: `/out_of_premise_diagnostics/84/ratio_classification/ratio`)_
This ratio belongs to an explicitly out-of-premise diagnostic; it is not presented as a guarantee violation.

## Independent audit classifications

- Audit: appendix_f_proof_ledger _(evidence: `/claim_results/0/audit`)_
- Model variant: paper_samplewise_literal _(evidence: `/claim_results/0/model_variant`)_
- Classification: contradicted _(evidence: `/claim_results/0/status`)_

- Audit: appendix_f_proof_ledger _(evidence: `/claim_results/1/audit`)_
- Model variant: appendix_inline_shift_literal _(evidence: `/claim_results/1/model_variant`)_
- Classification: contradicted _(evidence: `/claim_results/1/status`)_

- Audit: diminishing_returns _(evidence: `/claim_results/2/audit`)_
- Model variant: appendix_inline_shift_literal _(evidence: `/claim_results/2/model_variant`)_
- Classification: contradicted _(evidence: `/claim_results/2/status`)_

- Audit: greedy_guarantee_premise _(evidence: `/claim_results/3/audit`)_
- Model variant: appendix_inline_shift_literal _(evidence: `/claim_results/3/model_variant`)_
- Classification: contradicted _(evidence: `/claim_results/3/status`)_

- Audit: shift_boundary _(evidence: `/claim_results/4/audit`)_
- Model variant: modular_shift_candidate _(evidence: `/claim_results/4/model_variant`)_
- Classification: supported _(evidence: `/claim_results/4/status`)_

- Audit: objective_equivalence _(evidence: `/claim_results/5/audit`)_
- Model variant: paper_samplewise_literal _(evidence: `/claim_results/5/model_variant`)_
- Classification: contradicted _(evidence: `/claim_results/5/status`)_

## Exhaustive domains and ceilings

Every actual below is paired with its declared ceiling. A ceiling is a limit, not an equality target.
- Audit group: algorithm1 _(evidence: `/searches/0/id`)_
- Component: literal_algorithm1_audit _(evidence: `/searches/0/components/0/id`)_
- Actual: 1 _(evidence: `/searches/0/components/0/actual`)_
- Declared ceiling: 1 _(evidence: `/searches/0/components/0/declared_ceiling`)_

- Audit group: diminishing_returns _(evidence: `/searches/1/id`)_
- Component: appendix_e_witness_marginals _(evidence: `/searches/1/components/0/id`)_
- Actual: 2 _(evidence: `/searches/1/components/0/actual`)_
- Declared ceiling: 2 _(evidence: `/searches/1/components/0/declared_ceiling`)_
- Component: asymmetric_literal_diagnostic_primitives _(evidence: `/searches/1/components/1/id`)_
- Actual: 118428 _(evidence: `/searches/1/components/1/actual`)_
- Declared ceiling: 118428 _(evidence: `/searches/1/components/1/declared_ceiling`)_
- Component: symmetric_diminishing_return_primitives _(evidence: `/searches/1/components/2/id`)_
- Actual: 2861280 _(evidence: `/searches/1/components/2/actual`)_
- Declared ceiling: 2861280 _(evidence: `/searches/1/components/2/declared_ceiling`)_

- Audit group: greedy _(evidence: `/searches/2/id`)_
- Component: eq7_candidate_scores _(evidence: `/searches/2/components/0/id`)_
- Actual: 149175 _(evidence: `/searches/2/components/0/actual`)_
- Declared ceiling: 316983 _(evidence: `/searches/2/components/0/declared_ceiling`)_
- Component: eq7_terminal_paths _(evidence: `/searches/2/components/1/id`)_
- Actual: 42747 _(evidence: `/searches/2/components/1/actual`)_
- Declared ceiling: 210675 _(evidence: `/searches/2/components/1/declared_ceiling`)_
- Component: greedy_summary_classifications _(evidence: `/searches/2/components/2/id`)_
- Actual: 584604 _(evidence: `/searches/2/components/2/actual`)_
- Declared ceiling: 584604 _(evidence: `/searches/2/components/2/declared_ceiling`)_
- Component: optimum_subset_objective_values _(evidence: `/searches/2/components/3/id`)_
- Actual: 444870 _(evidence: `/searches/2/components/3/actual`)_
- Declared ceiling: 444870 _(evidence: `/searches/2/components/3/declared_ceiling`)_
- Component: premise_marginal_values _(evidence: `/searches/2/components/4/id`)_
- Actual: 1011330 _(evidence: `/searches/2/components/4/actual`)_
- Declared ceiling: 1011330 _(evidence: `/searches/2/components/4/declared_ceiling`)_
- Component: premise_submodularity_comparisons _(evidence: `/searches/2/components/5/id`)_
- Actual: 3394890 _(evidence: `/searches/2/components/5/actual`)_
- Declared ceiling: 3394890 _(evidence: `/searches/2/components/5/declared_ceiling`)_
- Component: premise_subset_values _(evidence: `/searches/2/components/6/id`)_
- Actual: 508500 _(evidence: `/searches/2/components/6/actual`)_
- Declared ceiling: 508500 _(evidence: `/searches/2/components/6/declared_ceiling`)_
- Component: true_marginal_candidate_lookups _(evidence: `/searches/2/components/7/id`)_
- Actual: 888066 _(evidence: `/searches/2/components/7/actual`)_
- Declared ceiling: 1901898 _(evidence: `/searches/2/components/7/declared_ceiling`)_
- Component: true_marginal_terminal_paths _(evidence: `/searches/2/components/8/id`)_
- Actual: 244674 _(evidence: `/searches/2/components/8/actual`)_
- Declared ceiling: 1264050 _(evidence: `/searches/2/components/8/declared_ceiling`)_

- Audit group: objective_equivalence _(evidence: `/searches/3/id`)_
- Component: objective_equivalence_objective_values _(evidence: `/searches/3/components/0/id`)_
- Actual: 52 _(evidence: `/searches/3/components/0/actual`)_
- Declared ceiling: 52 _(evidence: `/searches/3/components/0/declared_ceiling`)_

- Audit group: proof_ledger _(evidence: `/searches/4/id`)_
- Component: finite_appendix_f_conclusions _(evidence: `/searches/4/components/0/id`)_
- Actual: 1169208 _(evidence: `/searches/4/components/0/actual`)_
- Declared ceiling: 1169208 _(evidence: `/searches/4/components/0/declared_ceiling`)_
- Component: symbolic_appendix_f_conclusions _(evidence: `/searches/4/components/1/id`)_
- Actual: 84 _(evidence: `/searches/4/components/1/actual`)_
- Declared ceiling: 84 _(evidence: `/searches/4/components/1/declared_ceiling`)_

- Audit group: shifts _(evidence: `/searches/5/id`)_
- Component: rational_alpha_values _(evidence: `/searches/5/components/0/id`)_
- Actual: 1792 _(evidence: `/searches/5/components/0/actual`)_
- Declared ceiling: 1792 _(evidence: `/searches/5/components/0/declared_ceiling`)_
- Component: shift_marginal_score_values _(evidence: `/searches/5/components/1/id`)_
- Actual: 44870 _(evidence: `/searches/5/components/1/actual`)_
- Declared ceiling: 45213 _(evidence: `/searches/5/components/1/declared_ceiling`)_

## Appendix proof-ledger rows

The normalized nested conclusion records keep the literal, Appendix-inline, and modular candidate proof paths separate.
### `paper_samplewise_literal`

- Equation: 28 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/equation`)_
- Check: union_monotonicity_and_submodular_telescoping _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/0/status`)_
- Check: optimum_remainder_at_most_b_not_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/1/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/1/status`)_
- Equation: 29 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/1/equation`)_
- Check: true_marginal_argmax_and_defined_next_set _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/1/conclusions/0/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/1/conclusions/0/status`)_
- Equation: 30 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/2/equation`)_
- Check: combine_both_eq28_conclusions_and_eq29 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/2/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/2/conclusions/0/status`)_
- Equation: 31 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/3/equation`)_
- Check: defined_residual_algebra _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/3/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/3/conclusions/0/status`)_
- Equation: 32 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/4/equation`)_
- Check: divide_only_by_positive_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/4/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/4/conclusions/0/status`)_
- Equation: 33 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/5/equation`)_
- Check: residual_recurrence_definition _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/5/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/5/conclusions/0/status`)_
- Equation: 34 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/6/equation`)_
- Check: product_includes_k_equals_one_zero_factor _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/6/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/6/conclusions/0/status`)_
- Equation: 35 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/7/equation`)_
- Check: well_defined_positive_budget_product_bound _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/7/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/7/conclusions/0/status`)_
- Equation: 36 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/8/equation`)_
- Check: integer_budget_exponential_bound_and_log_domain _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/8/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/8/conclusions/0/status`)_
- Equation: 37 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/9/equation`)_
- Check: complete_chain_and_all_theorem_premises _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/9/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/9/conclusions/0/status`)_
- Equation: 38 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/10/equation`)_
- Check: ratio_transfer_from_repaired_objective _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/10/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/10/conclusions/0/status`)_

### `appendix_inline_shift_literal`

- Equation: 28 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/equation`)_
- Check: union_monotonicity_and_submodular_telescoping _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/0/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/0/status`)_
- Check: optimum_remainder_at_most_b_not_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/1/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/1/status`)_
- Equation: 29 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/1/equation`)_
- Check: true_marginal_argmax_and_defined_next_set _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/1/conclusions/0/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/1/conclusions/0/status`)_
- Equation: 30 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/2/equation`)_
- Check: combine_both_eq28_conclusions_and_eq29 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/2/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/2/conclusions/0/status`)_
- Equation: 31 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/3/equation`)_
- Check: defined_residual_algebra _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/3/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/3/conclusions/0/status`)_
- Equation: 32 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/4/equation`)_
- Check: divide_only_by_positive_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/4/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/4/conclusions/0/status`)_
- Equation: 33 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/5/equation`)_
- Check: residual_recurrence_definition _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/5/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/5/conclusions/0/status`)_
- Equation: 34 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/6/equation`)_
- Check: product_includes_k_equals_one_zero_factor _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/6/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/6/conclusions/0/status`)_
- Equation: 35 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/7/equation`)_
- Check: well_defined_positive_budget_product_bound _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/7/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/7/conclusions/0/status`)_
- Equation: 36 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/8/equation`)_
- Check: integer_budget_exponential_bound_and_log_domain _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/8/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/8/conclusions/0/status`)_
- Equation: 37 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/9/equation`)_
- Check: complete_chain_and_all_theorem_premises _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/9/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/9/conclusions/0/status`)_
- Equation: 38 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/10/equation`)_
- Check: ratio_transfer_from_repaired_objective _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/10/conclusions/0/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/10/conclusions/0/status`)_

### `modular_shift_candidate`

- Equation: 28 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/equation`)_
- Check: union_monotonicity_and_submodular_telescoping _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/0/status`)_
- Check: optimum_remainder_at_most_b_not_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/1/check_id`)_
- Status: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/1/status`)_
- Equation: 29 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/1/equation`)_
- Check: true_marginal_argmax_and_defined_next_set _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/1/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/1/conclusions/0/status`)_
- Equation: 30 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/2/equation`)_
- Check: combine_both_eq28_conclusions_and_eq29 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/2/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/2/conclusions/0/status`)_
- Equation: 31 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/3/equation`)_
- Check: defined_residual_algebra _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/3/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/3/conclusions/0/status`)_
- Equation: 32 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/4/equation`)_
- Check: divide_only_by_positive_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/4/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/4/conclusions/0/status`)_
- Equation: 33 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/5/equation`)_
- Check: residual_recurrence_definition _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/5/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/5/conclusions/0/status`)_
- Equation: 34 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/6/equation`)_
- Check: product_includes_k_equals_one_zero_factor _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/6/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/6/conclusions/0/status`)_
- Equation: 35 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/7/equation`)_
- Check: well_defined_positive_budget_product_bound _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/7/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/7/conclusions/0/status`)_
- Equation: 36 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/8/equation`)_
- Check: integer_budget_exponential_bound_and_log_domain _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/8/conclusions/0/check_id`)_
- Status: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/8/conclusions/0/status`)_
- Equation: 37 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/9/equation`)_
- Check: complete_chain_and_all_theorem_premises _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/9/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/9/conclusions/0/status`)_
- Equation: 38 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/10/equation`)_
- Check: ratio_transfer_from_repaired_objective _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/10/conclusions/0/check_id`)_
- Status: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/10/conclusions/0/status`)_

## Provenance and reviewed equations

- Pinned revision: arxiv:2606.12913v2 _(evidence: `/paper/revision`)_
- Source: https://arxiv.org/pdf/2606.12913v2 _(evidence: `/paper/source_url`)_
- PDF bytes: 683737 _(evidence: `/paper/pdf_byte_count`)_
- PDF digest: 26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 _(evidence: `/paper/pdf_sha256`)_
- Equation: Algorithm 1 _(evidence: `/transcriptions/records/0/equation`)_
- Reviewed expression: literal source lines 1-17 with PDF line wrapping normalized and no operational repair _(evidence: `/transcriptions/records/0/normalized_expression`)_
- Equation: 26 _(evidence: `/transcriptions/records/1/equation`)_
- Reviewed expression: Delta(x_i|S_hat) = alpha I_in(x_i) + alpha sum_{j=1}^{|S_hat|} eta + sum_{x_j in S_hat} g(D(x_i,x_j)) _(evidence: `/transcriptions/records/1/normalized_expression`)_
- Equation: 27 _(evidence: `/transcriptions/records/2/equation`)_
- Reviewed expression: eta >= (1/alpha) max_{x_i,x_j} |g(D(x_i,x_j))| _(evidence: `/transcriptions/records/2/normalized_expression`)_
- Equation: Appendix E inline _(evidence: `/transcriptions/records/3/equation`)_
- Reviewed expression: I_in_revised(x_i) = I_in(x_i) + sum_{j=1}^{|S_hat|} eta _(evidence: `/transcriptions/records/3/normalized_expression`)_
- Equation: 28-38 _(evidence: `/transcriptions/records/4/equation`)_
- Reviewed expression: literal Eq. 28-38 chain ending f(S_greedy) >= (1 - 1/e) f(S_star), without repairing its b-t or product steps _(evidence: `/transcriptions/records/4/normalized_expression`)_
- Equation: Appendix E literal-derived marginal _(evidence: `/transcriptions/records/5/equation`)_
- Reviewed expression: Delta_appendix(x|S) = alpha I_in(x) + 2 sum_{j in S} a_xj + alpha eta (2|S|+1) _(evidence: `/transcriptions/records/5/normalized_expression`)_
- Equation: Appendix E single-counted-derived marginal _(evidence: `/transcriptions/records/6/equation`)_
- Reviewed expression: Delta_single(x|S) = alpha I_in(x) + sum_{j in S} a_xj + alpha eta (2|S|+1) _(evidence: `/transcriptions/records/6/normalized_expression`)_
- Equation: 2 _(evidence: `/transcriptions/records/7/equation`)_
- Reviewed expression: w_i = alpha I_in(x_i); a_ij = g(D(x_i,x_j)) _(evidence: `/transcriptions/records/7/normalized_expression`)_
- Equation: 3 _(evidence: `/transcriptions/records/8/equation`)_
- Reviewed expression: maximize sum_{i in C} w_i + sum_{{i,j} subseteq C} a_ij subject to |C|=b _(evidence: `/transcriptions/records/8/normalized_expression`)_
- Equation: 4 _(evidence: `/transcriptions/records/9/equation`)_
- Reviewed expression: f(S) = sum_{x_i in S} [alpha I_in(x_i) + I_ex(x_i|S)] subject to |S|=b _(evidence: `/transcriptions/records/9/normalized_expression`)_
- Equation: 5 _(evidence: `/transcriptions/records/10/equation`)_
- Reviewed expression: I_ex(x_i|S) = sum_{x_j in S minus {x_i}} a_ij = sum_{x_j in S minus {x_i}} g(D(x_i,x_j)) _(evidence: `/transcriptions/records/10/normalized_expression`)_
- Equation: 6 _(evidence: `/transcriptions/records/11/equation`)_
- Reviewed expression: Delta_minus(v_i|G) = w_i + sum_{v_j in C minus {v_i}} a_ij _(evidence: `/transcriptions/records/11/normalized_expression`)_
- Equation: 7 _(evidence: `/transcriptions/records/12/equation`)_
- Reviewed expression: I(x_i|S) = Delta(x_i|S) = alpha I_in(x_i) + I_ex(x_i|S) _(evidence: `/transcriptions/records/12/normalized_expression`)_
- Equation: 8 _(evidence: `/transcriptions/records/13/equation`)_
- Reviewed expression: x_star in argmax_{x_i in T minus S_t} I(x_i|S_t); S_{t+1} = S_t union {x_star} _(evidence: `/transcriptions/records/13/normalized_expression`)_
- Equation: 10-11 _(evidence: `/transcriptions/records/14/equation`)_
- Reviewed expression: Delta(x|A) >= Delta(x|B), where Delta(x|A) = f(A union {x}) - f(A) _(evidence: `/transcriptions/records/14/normalized_expression`)_
- Equation: 12-14 _(evidence: `/transcriptions/records/15/equation`)_
- Reviewed expression: Delta_A = alpha I_in + sum_A g; Delta_B = alpha I_in + sum_B g; Delta_A - Delta_B = -sum_{B minus A} g >= 0 _(evidence: `/transcriptions/records/15/normalized_expression`)_

## Limitations

Bounded enumeration can refute but cannot prove arbitrary-real universal claims. No released implementation resolves the paper's edge-counting ambiguity or Appendix shift ambiguity.

## Unavailable empirical claims

- Status: unavailable _(evidence: `/unavailable_claims/0/status`)_
- Boundary: Paper-reported CIFAR-10/100 and ImageNet-1k accuracy, training-time, and acceleration values were not recomputed. _(evidence: `/unavailable_claims/0/reason`)_
- Status: unavailable _(evidence: `/unavailable_claims/1/status`)_
- Boundary: Paper-reported detection and segmentation experiments were not recomputed. _(evidence: `/unavailable_claims/1/reason`)_

## Attribution and licenses

Seven-author paper attribution and adaptation details are in `NOTICE.md`. Original executable code and schema are covered by `LICENSE`; transcriptions and evidence are covered by `LICENSES/CC-BY-NC-SA-4.0.txt`.
- Paper asset license: CC BY-NC-SA 4.0 _(evidence: `/paper/license`)_

## Complete display-pointer ledger

- Displayed evidence: The paper casts dataset pruning as a graph problem with node weights for intrinsic importance and edge weights for extrinsic diversity/interaction, yielding a Maximum Weight Clique formulation (Section 3.3). _(evidence: `/target_claims/0`)_
- Displayed evidence: Under mild conditions, the unified objective becomes submodular and admits a greedy approximation guarantee (Section 3.6; Appendix F). _(evidence: `/target_claims/1`)_
- Displayed evidence: arxiv:2606.12913v2 _(evidence: `/paper/revision`)_
- Displayed evidence: https://arxiv.org/pdf/2606.12913v2 _(evidence: `/paper/source_url`)_
- Displayed evidence: 683737 _(evidence: `/paper/pdf_byte_count`)_
- Displayed evidence: 26ce80e8d347340e0055f2bcf061b6b3e29489fc68a85b8d5711e12cc9da5090 _(evidence: `/paper/pdf_sha256`)_
- Displayed evidence: CC BY-NC-SA 4.0 _(evidence: `/paper/license`)_
- Displayed evidence: 9ef7b07d8cb7349215aa5f780ec346a9024c9396 _(evidence: `/source_revision`)_
- Displayed evidence: cpu _(evidence: `/environment/compute`)_
- Displayed evidence: False _(evidence: `/environment/network_used`)_
- Displayed evidence: 0/1 _(evidence: `/environment/paid_api_cost_usd`)_
- Displayed evidence: 11464573 _(evidence: `/commands/0/actual`)_
- Displayed evidence: 13833860 _(evidence: `/commands/0/ceiling`)_
- Displayed evidence: Algorithm 1 _(evidence: `/transcriptions/records/0/equation`)_
- Displayed evidence: Algorithm 1 _(evidence: `/transcriptions/records/0/section`)_
- Displayed evidence: literal source lines 1-17 with PDF line wrapping normalized and no operational repair _(evidence: `/transcriptions/records/0/normalized_expression`)_
- Displayed evidence: 26 _(evidence: `/transcriptions/records/1/equation`)_
- Displayed evidence: Appendix E Maintain Monotonicity _(evidence: `/transcriptions/records/1/section`)_
- Displayed evidence: Delta(x_i|S_hat) = alpha I_in(x_i) + alpha sum_{j=1}^{|S_hat|} eta + sum_{x_j in S_hat} g(D(x_i,x_j)) _(evidence: `/transcriptions/records/1/normalized_expression`)_
- Displayed evidence: 27 _(evidence: `/transcriptions/records/2/equation`)_
- Displayed evidence: Appendix E Maintain Monotonicity _(evidence: `/transcriptions/records/2/section`)_
- Displayed evidence: eta >= (1/alpha) max_{x_i,x_j} |g(D(x_i,x_j))| _(evidence: `/transcriptions/records/2/normalized_expression`)_
- Displayed evidence: Appendix E inline _(evidence: `/transcriptions/records/3/equation`)_
- Displayed evidence: Appendix E Maintain Monotonicity _(evidence: `/transcriptions/records/3/section`)_
- Displayed evidence: I_in_revised(x_i) = I_in(x_i) + sum_{j=1}^{|S_hat|} eta _(evidence: `/transcriptions/records/3/normalized_expression`)_
- Displayed evidence: 28-38 _(evidence: `/transcriptions/records/4/equation`)_
- Displayed evidence: Appendix F Proof of the Greedy Approximation Guarantee _(evidence: `/transcriptions/records/4/section`)_
- Displayed evidence: literal Eq. 28-38 chain ending f(S_greedy) >= (1 - 1/e) f(S_star), without repairing its b-t or product steps _(evidence: `/transcriptions/records/4/normalized_expression`)_
- Displayed evidence: Appendix E literal-derived marginal _(evidence: `/transcriptions/records/5/equation`)_
- Displayed evidence: Approved design derivation from Appendix E and Eqs. 4-5 _(evidence: `/transcriptions/records/5/section`)_
- Displayed evidence: Delta_appendix(x|S) = alpha I_in(x) + 2 sum_{j in S} a_xj + alpha eta (2|S|+1) _(evidence: `/transcriptions/records/5/normalized_expression`)_
- Displayed evidence: Appendix E single-counted-derived marginal _(evidence: `/transcriptions/records/6/equation`)_
- Displayed evidence: Approved design derivation for the repaired single-counted objective _(evidence: `/transcriptions/records/6/section`)_
- Displayed evidence: Delta_single(x|S) = alpha I_in(x) + sum_{j in S} a_xj + alpha eta (2|S|+1) _(evidence: `/transcriptions/records/6/normalized_expression`)_
- Displayed evidence: 2 _(evidence: `/transcriptions/records/7/equation`)_
- Displayed evidence: 3.3 Rethinking Pruning from a Graph Perspective _(evidence: `/transcriptions/records/7/section`)_
- Displayed evidence: w_i = alpha I_in(x_i); a_ij = g(D(x_i,x_j)) _(evidence: `/transcriptions/records/7/normalized_expression`)_
- Displayed evidence: 3 _(evidence: `/transcriptions/records/8/equation`)_
- Displayed evidence: 3.3 Maximum Weight Clique Formulation _(evidence: `/transcriptions/records/8/section`)_
- Displayed evidence: maximize sum_{i in C} w_i + sum_{{i,j} subseteq C} a_ij subject to |C|=b _(evidence: `/transcriptions/records/8/normalized_expression`)_
- Displayed evidence: 4 _(evidence: `/transcriptions/records/9/equation`)_
- Displayed evidence: 3.3 Sample-wise Reformulation _(evidence: `/transcriptions/records/9/section`)_
- Displayed evidence: f(S) = sum_{x_i in S} [alpha I_in(x_i) + I_ex(x_i|S)] subject to |S|=b _(evidence: `/transcriptions/records/9/normalized_expression`)_
- Displayed evidence: 5 _(evidence: `/transcriptions/records/10/equation`)_
- Displayed evidence: 3.3 Sample-wise Reformulation _(evidence: `/transcriptions/records/10/section`)_
- Displayed evidence: I_ex(x_i|S) = sum_{x_j in S minus {x_i}} a_ij = sum_{x_j in S minus {x_i}} g(D(x_i,x_j)) _(evidence: `/transcriptions/records/10/normalized_expression`)_
- Displayed evidence: 6 _(evidence: `/transcriptions/records/11/equation`)_
- Displayed evidence: 3.4 Greedy Selection with Unified Importance _(evidence: `/transcriptions/records/11/section`)_
- Displayed evidence: Delta_minus(v_i|G) = w_i + sum_{v_j in C minus {v_i}} a_ij _(evidence: `/transcriptions/records/11/normalized_expression`)_
- Displayed evidence: 7 _(evidence: `/transcriptions/records/12/equation`)_
- Displayed evidence: 3.4 Unified Importance _(evidence: `/transcriptions/records/12/section`)_
- Displayed evidence: I(x_i|S) = Delta(x_i|S) = alpha I_in(x_i) + I_ex(x_i|S) _(evidence: `/transcriptions/records/12/normalized_expression`)_
- Displayed evidence: 8 _(evidence: `/transcriptions/records/13/equation`)_
- Displayed evidence: 3.4 Greedy Selection Strategy _(evidence: `/transcriptions/records/13/section`)_
- Displayed evidence: x_star in argmax_{x_i in T minus S_t} I(x_i|S_t); S_{t+1} = S_t union {x_star} _(evidence: `/transcriptions/records/13/normalized_expression`)_
- Displayed evidence: 10-11 _(evidence: `/transcriptions/records/14/equation`)_
- Displayed evidence: 3.6 Definition 3.3 _(evidence: `/transcriptions/records/14/section`)_
- Displayed evidence: Delta(x|A) >= Delta(x|B), where Delta(x|A) = f(A union {x}) - f(A) _(evidence: `/transcriptions/records/14/normalized_expression`)_
- Displayed evidence: 12-14 _(evidence: `/transcriptions/records/15/equation`)_
- Displayed evidence: 3.6 Lemma 3.4 proof _(evidence: `/transcriptions/records/15/section`)_
- Displayed evidence: Delta_A = alpha I_in + sum_A g; Delta_B = alpha I_in + sum_B g; Delta_A - Delta_B = -sum_{B minus A} g >= 0 _(evidence: `/transcriptions/records/15/normalized_expression`)_
- Displayed evidence: algorithm1 _(evidence: `/searches/0/id`)_
- Displayed evidence: symbolic _(evidence: `/searches/0/evidence_kind`)_
- Displayed evidence: True _(evidence: `/searches/0/completed`)_
- Displayed evidence: literal_algorithm1_audit _(evidence: `/searches/0/components/0/id`)_
- Displayed evidence: 1 _(evidence: `/searches/0/components/0/actual`)_
- Displayed evidence: 1 _(evidence: `/searches/0/components/0/declared_ceiling`)_
- Displayed evidence: diminishing_returns _(evidence: `/searches/1/id`)_
- Displayed evidence: exhaustive_finite _(evidence: `/searches/1/evidence_kind`)_
- Displayed evidence: True _(evidence: `/searches/1/completed`)_
- Displayed evidence: appendix_e_witness_marginals _(evidence: `/searches/1/components/0/id`)_
- Displayed evidence: 2 _(evidence: `/searches/1/components/0/actual`)_
- Displayed evidence: 2 _(evidence: `/searches/1/components/0/declared_ceiling`)_
- Displayed evidence: asymmetric_literal_diagnostic_primitives _(evidence: `/searches/1/components/1/id`)_
- Displayed evidence: 118428 _(evidence: `/searches/1/components/1/actual`)_
- Displayed evidence: 118428 _(evidence: `/searches/1/components/1/declared_ceiling`)_
- Displayed evidence: symmetric_diminishing_return_primitives _(evidence: `/searches/1/components/2/id`)_
- Displayed evidence: 2861280 _(evidence: `/searches/1/components/2/actual`)_
- Displayed evidence: 2861280 _(evidence: `/searches/1/components/2/declared_ceiling`)_
- Displayed evidence: greedy _(evidence: `/searches/2/id`)_
- Displayed evidence: exhaustive_finite _(evidence: `/searches/2/evidence_kind`)_
- Displayed evidence: True _(evidence: `/searches/2/completed`)_
- Displayed evidence: eq7_candidate_scores _(evidence: `/searches/2/components/0/id`)_
- Displayed evidence: 149175 _(evidence: `/searches/2/components/0/actual`)_
- Displayed evidence: 316983 _(evidence: `/searches/2/components/0/declared_ceiling`)_
- Displayed evidence: eq7_terminal_paths _(evidence: `/searches/2/components/1/id`)_
- Displayed evidence: 42747 _(evidence: `/searches/2/components/1/actual`)_
- Displayed evidence: 210675 _(evidence: `/searches/2/components/1/declared_ceiling`)_
- Displayed evidence: greedy_summary_classifications _(evidence: `/searches/2/components/2/id`)_
- Displayed evidence: 584604 _(evidence: `/searches/2/components/2/actual`)_
- Displayed evidence: 584604 _(evidence: `/searches/2/components/2/declared_ceiling`)_
- Displayed evidence: optimum_subset_objective_values _(evidence: `/searches/2/components/3/id`)_
- Displayed evidence: 444870 _(evidence: `/searches/2/components/3/actual`)_
- Displayed evidence: 444870 _(evidence: `/searches/2/components/3/declared_ceiling`)_
- Displayed evidence: premise_marginal_values _(evidence: `/searches/2/components/4/id`)_
- Displayed evidence: 1011330 _(evidence: `/searches/2/components/4/actual`)_
- Displayed evidence: 1011330 _(evidence: `/searches/2/components/4/declared_ceiling`)_
- Displayed evidence: premise_submodularity_comparisons _(evidence: `/searches/2/components/5/id`)_
- Displayed evidence: 3394890 _(evidence: `/searches/2/components/5/actual`)_
- Displayed evidence: 3394890 _(evidence: `/searches/2/components/5/declared_ceiling`)_
- Displayed evidence: premise_subset_values _(evidence: `/searches/2/components/6/id`)_
- Displayed evidence: 508500 _(evidence: `/searches/2/components/6/actual`)_
- Displayed evidence: 508500 _(evidence: `/searches/2/components/6/declared_ceiling`)_
- Displayed evidence: true_marginal_candidate_lookups _(evidence: `/searches/2/components/7/id`)_
- Displayed evidence: 888066 _(evidence: `/searches/2/components/7/actual`)_
- Displayed evidence: 1901898 _(evidence: `/searches/2/components/7/declared_ceiling`)_
- Displayed evidence: true_marginal_terminal_paths _(evidence: `/searches/2/components/8/id`)_
- Displayed evidence: 244674 _(evidence: `/searches/2/components/8/actual`)_
- Displayed evidence: 1264050 _(evidence: `/searches/2/components/8/declared_ceiling`)_
- Displayed evidence: objective_equivalence _(evidence: `/searches/3/id`)_
- Displayed evidence: symbolic _(evidence: `/searches/3/evidence_kind`)_
- Displayed evidence: True _(evidence: `/searches/3/completed`)_
- Displayed evidence: objective_equivalence_objective_values _(evidence: `/searches/3/components/0/id`)_
- Displayed evidence: 52 _(evidence: `/searches/3/components/0/actual`)_
- Displayed evidence: 52 _(evidence: `/searches/3/components/0/declared_ceiling`)_
- Displayed evidence: proof_ledger _(evidence: `/searches/4/id`)_
- Displayed evidence: exhaustive_finite _(evidence: `/searches/4/evidence_kind`)_
- Displayed evidence: True _(evidence: `/searches/4/completed`)_
- Displayed evidence: finite_appendix_f_conclusions _(evidence: `/searches/4/components/0/id`)_
- Displayed evidence: 1169208 _(evidence: `/searches/4/components/0/actual`)_
- Displayed evidence: 1169208 _(evidence: `/searches/4/components/0/declared_ceiling`)_
- Displayed evidence: symbolic_appendix_f_conclusions _(evidence: `/searches/4/components/1/id`)_
- Displayed evidence: 84 _(evidence: `/searches/4/components/1/actual`)_
- Displayed evidence: 84 _(evidence: `/searches/4/components/1/declared_ceiling`)_
- Displayed evidence: shifts _(evidence: `/searches/5/id`)_
- Displayed evidence: non_exhaustive _(evidence: `/searches/5/evidence_kind`)_
- Displayed evidence: True _(evidence: `/searches/5/completed`)_
- Displayed evidence: rational_alpha_values _(evidence: `/searches/5/components/0/id`)_
- Displayed evidence: 1792 _(evidence: `/searches/5/components/0/actual`)_
- Displayed evidence: 1792 _(evidence: `/searches/5/components/0/declared_ceiling`)_
- Displayed evidence: shift_marginal_score_values _(evidence: `/searches/5/components/1/id`)_
- Displayed evidence: 44870 _(evidence: `/searches/5/components/1/actual`)_
- Displayed evidence: 45213 _(evidence: `/searches/5/components/1/declared_ceiling`)_
- Displayed evidence: evidence/witnesses/8ab3857832c3bc9f.json _(evidence: `/witnesses/0/artifact_path`)_
- Displayed evidence: 07cf661c91d70fc04700d91213d3c351bd3a3376785736ac34ce9ad750318aa7 _(evidence: `/witnesses/0/artifact_sha256`)_
- Displayed evidence: 2 > 1 _(evidence: `/witnesses/0/classification/comparison`)_
- Displayed evidence: False _(evidence: `/witnesses/0/classification/eq28a_counterexample`)_
- Displayed evidence: contradicted _(evidence: `/witnesses/0/classification/eq28b_cardinality_bound`)_
- Displayed evidence: False _(evidence: `/witnesses/0/classification/theorem_counterexample`)_
- Displayed evidence: symbolic _(evidence: `/witnesses/0/evidence_kind`)_
- Displayed evidence: cardinality-b-minus-t _(evidence: `/witnesses/0/id`)_
- Displayed evidence: 2 _(evidence: `/witnesses/0/inputs/budget`)_
- Displayed evidence: 1 _(evidence: `/witnesses/0/inputs/iteration`)_
- Displayed evidence: b _(evidence: `/witnesses/0/inputs/s_star/0`)_
- Displayed evidence: c _(evidence: `/witnesses/0/inputs/s_star/1`)_
- Displayed evidence: a _(evidence: `/witnesses/0/inputs/s_t/0`)_
- Displayed evidence: a _(evidence: `/witnesses/0/inputs/vertices/0`)_
- Displayed evidence: b _(evidence: `/witnesses/0/inputs/vertices/1`)_
- Displayed evidence: c _(evidence: `/witnesses/0/inputs/vertices/2`)_
- Displayed evidence: none _(evidence: `/witnesses/0/inputs/weight_assumptions`)_
- Displayed evidence: 1 _(evidence: `/witnesses/0/intermediate_values/b_minus_t`)_
- Displayed evidence: 2 _(evidence: `/witnesses/0/intermediate_values/remainder_cardinality`)_
- Displayed evidence: b _(evidence: `/witnesses/0/intermediate_values/s_star_minus_s_t/0`)_
- Displayed evidence: c _(evidence: `/witnesses/0/intermediate_values/s_star_minus_s_t/1`)_
- Displayed evidence: optimum_remainder_cardinality_exceeds_b_minus_t _(evidence: `/witnesses/0/property`)_
- Displayed evidence: evidence/witnesses/5068c5858180bb4c.json _(evidence: `/witnesses/1/artifact_path`)_
- Displayed evidence: 5b06099fe7bce9f7bfd110755c0ea0f8b9532b0e18499d8689114d3ece5899d6 _(evidence: `/witnesses/1/artifact_sha256`)_
- Displayed evidence: symbolic _(evidence: `/witnesses/1/evidence_kind`)_
- Displayed evidence: graph=n=2;vw=0/1,0/1;ew=0/1::variant=appendix_inline_shift_literal::alpha=1/1::eta=1/1::diagnostic=appendix-minimal _(evidence: `/witnesses/1/id`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/1/inputs/alpha`)_
- Displayed evidence: v1 _(evidence: `/witnesses/1/inputs/b/0`)_
- Displayed evidence: v0 _(evidence: `/witnesses/1/inputs/candidate`)_
- Displayed evidence: 0/1 _(evidence: `/witnesses/1/inputs/edge_weights/0`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/1/inputs/eta`)_
- Displayed evidence: 0/1 _(evidence: `/witnesses/1/inputs/vertex_weights/0`)_
- Displayed evidence: 0/1 _(evidence: `/witnesses/1/inputs/vertex_weights/1`)_
- Displayed evidence: v0 _(evidence: `/witnesses/1/inputs/vertices/0`)_
- Displayed evidence: v1 _(evidence: `/witnesses/1/inputs/vertices/1`)_
- Displayed evidence: -2/1 _(evidence: `/witnesses/1/intermediate_values/difference`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/1/intermediate_values/marginal_empty`)_
- Displayed evidence: 3/1 _(evidence: `/witnesses/1/intermediate_values/marginal_y`)_
- Displayed evidence: False _(evidence: `/witnesses/1/minimality_checks/one_vertex_strict_chain_exists`)_
- Displayed evidence: True _(evidence: `/witnesses/1/minimality_checks/two_vertices_required`)_
- Displayed evidence: appendix_inline_shift_literal _(evidence: `/witnesses/1/model_variant`)_
- Displayed evidence: appendix_inline_shift_diminishing_returns _(evidence: `/witnesses/1/property`)_
- Displayed evidence: evidence/witnesses/5a4759c10574d713.json _(evidence: `/witnesses/2/artifact_path`)_
- Displayed evidence: 4dda75368f4b252582e0d7499fd0d69c61dbf0d35f0e966e2eb5ae2d18ded97b _(evidence: `/witnesses/2/artifact_sha256`)_
- Displayed evidence: n=2;selected=v0,v1;vw=0/1,0/1;ew=1/1 _(evidence: `/witnesses/2/case_id`)_
- Displayed evidence: 1 _(evidence: `/witnesses/2/comparison/mwcp_edge_coefficient`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/2/comparison/paper_mwcp`)_
- Displayed evidence: 2/1 _(evidence: `/witnesses/2/comparison/paper_samplewise_literal`)_
- Displayed evidence: 2 _(evidence: `/witnesses/2/comparison/samplewise_edge_coefficient`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/2/comparison/samplewise_minus_mwcp`)_
- Displayed evidence: objective-equivalence-n2-edge1 _(evidence: `/witnesses/2/id`)_
- Displayed evidence: v0 _(evidence: `/witnesses/2/interactions/0/0`)_
- Displayed evidence: v1 _(evidence: `/witnesses/2/interactions/0/1`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/2/interactions/0/2`)_
- Displayed evidence: v1 _(evidence: `/witnesses/2/interactions/1/0`)_
- Displayed evidence: v0 _(evidence: `/witnesses/2/interactions/1/1`)_
- Displayed evidence: 1/1 _(evidence: `/witnesses/2/interactions/1/2`)_
- Displayed evidence: paper_samplewise_literal _(evidence: `/witnesses/2/model_variant`)_
- Displayed evidence: paper_mwcp_vs_paper_samplewise_literal _(evidence: `/witnesses/2/property`)_
- Displayed evidence: v0 _(evidence: `/witnesses/2/selected/0`)_
- Displayed evidence: v1 _(evidence: `/witnesses/2/selected/1`)_
- Displayed evidence: v0 _(evidence: `/witnesses/2/vertex_weights/0/0`)_
- Displayed evidence: 0/1 _(evidence: `/witnesses/2/vertex_weights/0/1`)_
- Displayed evidence: v1 _(evidence: `/witnesses/2/vertex_weights/1/0`)_
- Displayed evidence: 0/1 _(evidence: `/witnesses/2/vertex_weights/1/1`)_
- Displayed evidence: v0 _(evidence: `/witnesses/2/vertices/0`)_
- Displayed evidence: v1 _(evidence: `/witnesses/2/vertices/1`)_
- Displayed evidence: appendix_f_proof_ledger _(evidence: `/claim_results/0/audit`)_
- Displayed evidence: paper_samplewise_literal _(evidence: `/claim_results/0/model_variant`)_
- Displayed evidence: symbolic _(evidence: `/claim_results/0/evidence_kind`)_
- Displayed evidence: contradicted _(evidence: `/claim_results/0/status`)_
- Displayed evidence: appendix_f_proof_ledger _(evidence: `/claim_results/1/audit`)_
- Displayed evidence: appendix_inline_shift_literal _(evidence: `/claim_results/1/model_variant`)_
- Displayed evidence: symbolic _(evidence: `/claim_results/1/evidence_kind`)_
- Displayed evidence: contradicted _(evidence: `/claim_results/1/status`)_
- Displayed evidence: diminishing_returns _(evidence: `/claim_results/2/audit`)_
- Displayed evidence: appendix_inline_shift_literal _(evidence: `/claim_results/2/model_variant`)_
- Displayed evidence: symbolic _(evidence: `/claim_results/2/evidence_kind`)_
- Displayed evidence: contradicted _(evidence: `/claim_results/2/status`)_
- Displayed evidence: greedy_guarantee_premise _(evidence: `/claim_results/3/audit`)_
- Displayed evidence: appendix_inline_shift_literal _(evidence: `/claim_results/3/model_variant`)_
- Displayed evidence: symbolic _(evidence: `/claim_results/3/evidence_kind`)_
- Displayed evidence: contradicted _(evidence: `/claim_results/3/status`)_
- Displayed evidence: shift_boundary _(evidence: `/claim_results/4/audit`)_
- Displayed evidence: modular_shift_candidate _(evidence: `/claim_results/4/model_variant`)_
- Displayed evidence: exhaustive_finite _(evidence: `/claim_results/4/evidence_kind`)_
- Displayed evidence: supported _(evidence: `/claim_results/4/status`)_
- Displayed evidence: objective_equivalence _(evidence: `/claim_results/5/audit`)_
- Displayed evidence: paper_samplewise_literal _(evidence: `/claim_results/5/model_variant`)_
- Displayed evidence: symbolic _(evidence: `/claim_results/5/evidence_kind`)_
- Displayed evidence: contradicted _(evidence: `/claim_results/5/status`)_
- Displayed evidence: cifar-imagenet-training-results _(evidence: `/unavailable_claims/0/id`)_
- Displayed evidence: unavailable _(evidence: `/unavailable_claims/0/status`)_
- Displayed evidence: Paper-reported CIFAR-10/100 and ImageNet-1k accuracy, training-time, and acceleration values were not recomputed. _(evidence: `/unavailable_claims/0/reason`)_
- Displayed evidence: detection-segmentation-results _(evidence: `/unavailable_claims/1/id`)_
- Displayed evidence: unavailable _(evidence: `/unavailable_claims/1/status`)_
- Displayed evidence: Paper-reported detection and segmentation experiments were not recomputed. _(evidence: `/unavailable_claims/1/reason`)_
- Displayed evidence: graph=n=3;vw=0/1,0/1,0/1;ew=-1/1,-1/1,0/1::variant=appendix_inline_shift_literal::alpha=1/1::eta=1/1::budget=2::selector=paper_eq7_score_greedy::result=canonical::path=v0,v1 _(evidence: `/out_of_premise_diagnostics/84/id`)_
- Displayed evidence: defined_positive_optimum _(evidence: `/out_of_premise_diagnostics/84/ratio_classification/status`)_
- Displayed evidence: 1/2 _(evidence: `/out_of_premise_diagnostics/84/ratio_classification/ratio`)_
- Displayed evidence: 28 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/equation`)_
- Displayed evidence: union_monotonicity_and_submodular_telescoping _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/0/status`)_
- Displayed evidence: optimum_remainder_at_most_b_not_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/1/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/0/conclusions/1/status`)_
- Displayed evidence: 29 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/1/equation`)_
- Displayed evidence: true_marginal_argmax_and_defined_next_set _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/1/conclusions/0/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/1/conclusions/0/status`)_
- Displayed evidence: 30 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/2/equation`)_
- Displayed evidence: combine_both_eq28_conclusions_and_eq29 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/2/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/2/conclusions/0/status`)_
- Displayed evidence: 31 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/3/equation`)_
- Displayed evidence: defined_residual_algebra _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/3/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/3/conclusions/0/status`)_
- Displayed evidence: 32 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/4/equation`)_
- Displayed evidence: divide_only_by_positive_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/4/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/4/conclusions/0/status`)_
- Displayed evidence: 33 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/5/equation`)_
- Displayed evidence: residual_recurrence_definition _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/5/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/5/conclusions/0/status`)_
- Displayed evidence: 34 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/6/equation`)_
- Displayed evidence: product_includes_k_equals_one_zero_factor _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/6/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/6/conclusions/0/status`)_
- Displayed evidence: 35 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/7/equation`)_
- Displayed evidence: well_defined_positive_budget_product_bound _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/7/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/7/conclusions/0/status`)_
- Displayed evidence: 36 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/8/equation`)_
- Displayed evidence: integer_budget_exponential_bound_and_log_domain _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/8/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/8/conclusions/0/status`)_
- Displayed evidence: 37 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/9/equation`)_
- Displayed evidence: complete_chain_and_all_theorem_premises _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/9/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/9/conclusions/0/status`)_
- Displayed evidence: 38 _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/10/equation`)_
- Displayed evidence: ratio_transfer_from_repaired_objective _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/10/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/paper_samplewise_literal/10/conclusions/0/status`)_
- Displayed evidence: 28 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/equation`)_
- Displayed evidence: union_monotonicity_and_submodular_telescoping _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/0/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/0/status`)_
- Displayed evidence: optimum_remainder_at_most_b_not_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/1/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/0/conclusions/1/status`)_
- Displayed evidence: 29 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/1/equation`)_
- Displayed evidence: true_marginal_argmax_and_defined_next_set _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/1/conclusions/0/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/1/conclusions/0/status`)_
- Displayed evidence: 30 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/2/equation`)_
- Displayed evidence: combine_both_eq28_conclusions_and_eq29 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/2/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/2/conclusions/0/status`)_
- Displayed evidence: 31 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/3/equation`)_
- Displayed evidence: defined_residual_algebra _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/3/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/3/conclusions/0/status`)_
- Displayed evidence: 32 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/4/equation`)_
- Displayed evidence: divide_only_by_positive_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/4/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/4/conclusions/0/status`)_
- Displayed evidence: 33 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/5/equation`)_
- Displayed evidence: residual_recurrence_definition _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/5/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/5/conclusions/0/status`)_
- Displayed evidence: 34 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/6/equation`)_
- Displayed evidence: product_includes_k_equals_one_zero_factor _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/6/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/6/conclusions/0/status`)_
- Displayed evidence: 35 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/7/equation`)_
- Displayed evidence: well_defined_positive_budget_product_bound _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/7/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/7/conclusions/0/status`)_
- Displayed evidence: 36 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/8/equation`)_
- Displayed evidence: integer_budget_exponential_bound_and_log_domain _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/8/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/8/conclusions/0/status`)_
- Displayed evidence: 37 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/9/equation`)_
- Displayed evidence: complete_chain_and_all_theorem_premises _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/9/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/9/conclusions/0/status`)_
- Displayed evidence: 38 _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/10/equation`)_
- Displayed evidence: ratio_transfer_from_repaired_objective _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/10/conclusions/0/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/appendix_inline_shift_literal/10/conclusions/0/status`)_
- Displayed evidence: 28 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/equation`)_
- Displayed evidence: union_monotonicity_and_submodular_telescoping _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/0/status`)_
- Displayed evidence: optimum_remainder_at_most_b_not_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/1/check_id`)_
- Displayed evidence: contradicted _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/0/conclusions/1/status`)_
- Displayed evidence: 29 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/1/equation`)_
- Displayed evidence: true_marginal_argmax_and_defined_next_set _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/1/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/1/conclusions/0/status`)_
- Displayed evidence: 30 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/2/equation`)_
- Displayed evidence: combine_both_eq28_conclusions_and_eq29 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/2/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/2/conclusions/0/status`)_
- Displayed evidence: 31 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/3/equation`)_
- Displayed evidence: defined_residual_algebra _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/3/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/3/conclusions/0/status`)_
- Displayed evidence: 32 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/4/equation`)_
- Displayed evidence: divide_only_by_positive_b_minus_t _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/4/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/4/conclusions/0/status`)_
- Displayed evidence: 33 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/5/equation`)_
- Displayed evidence: residual_recurrence_definition _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/5/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/5/conclusions/0/status`)_
- Displayed evidence: 34 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/6/equation`)_
- Displayed evidence: product_includes_k_equals_one_zero_factor _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/6/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/6/conclusions/0/status`)_
- Displayed evidence: 35 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/7/equation`)_
- Displayed evidence: well_defined_positive_budget_product_bound _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/7/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/7/conclusions/0/status`)_
- Displayed evidence: 36 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/8/equation`)_
- Displayed evidence: integer_budget_exponential_bound_and_log_domain _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/8/conclusions/0/check_id`)_
- Displayed evidence: supported _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/8/conclusions/0/status`)_
- Displayed evidence: 37 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/9/equation`)_
- Displayed evidence: complete_chain_and_all_theorem_premises _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/9/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/9/conclusions/0/status`)_
- Displayed evidence: 38 _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/10/equation`)_
- Displayed evidence: ratio_transfer_from_repaired_objective _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/10/conclusions/0/check_id`)_
- Displayed evidence: not_applicable _(evidence: `/proof_ledger/symbolic/ledgers/modular_shift_candidate/10/conclusions/0/status`)_
