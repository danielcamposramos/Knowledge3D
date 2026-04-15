# HS Math Cluster 3 — Statistics, Probability, Combinatorics, Discrete, Financial, Applied

**Phase:** 7.A.2
**Cluster:** 3 of 3
**Format:** bullet dialect (parsed by `parse_cluster1_bullets`)
**Canonical id prefixes:** `formula_`, `identity_`, `theorem_`, `rule_`, `concept_`, `method_`
**RPN palette:** STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK
**Surface form languages:** en, pt, es, fr, de, it, ja, zh, ru
**Symlink namespaces:** `star.letter.*`, `star.symbol.*`, `star.constant.*`, `star.concept.*`

Scope: high-school descriptive statistics, probability, combinatorics, set theory and propositional logic basics, financial math, applied physics + chemistry formulas, units. Every star is meaning-first (language-agnostic) with bidirectional symlinks to Phase 7.A.1 letter / symbol / constant stars.

## Cluster 3.A — Descriptive Statistics

#### formula_arithmetic_mean
- **canonical_id**: `formula_arithmetic_mean`
- **is_a**: `formula_central_tendency`
- **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.sum][TPACK 1][RECALL n][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE mean]`
- **symlinks**: `star.letter.x, star.letter.n, star.symbol.sum, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "x̄ = (Σ xᵢ) / n"
  - pt: "x̄ = (Σ xᵢ) / n"
  - es: "x̄ = (Σ xᵢ) / n"
  - fr: "x̄ = (Σ xᵢ) / n"
  - de: "x̄ = (Σ xᵢ) / n"
  - it: "x̄ = (Σ xᵢ) / n"
  - ja: "x̄ = (Σ xᵢ) / n"
  - zh: "x̄ = (Σ xᵢ) / n"
  - ru: "x̄ = (Σ xᵢ) / n"

#### formula_weighted_mean
- **canonical_id**: `formula_weighted_mean`
- **is_a**: `formula_central_tendency`
- **rpn_sketch**: `[RECALL w_i][RECALL x_i][TMUL][GALAXY_LOOKUP star.symbol.sum][TPACK 1][RECALL w_i][GALAXY_LOOKUP star.symbol.sum][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE wmean]`
- **symlinks**: `star.letter.w, star.letter.x, star.symbol.sum, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "x̄_w = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - pt: "x̄_p = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - es: "x̄_p = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - fr: "x̄_p = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - de: "x̄_g = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - it: "x̄_p = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - ja: "加重平均 = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - zh: "加权平均 = (Σ wᵢxᵢ) / (Σ wᵢ)"
  - ru: "x̄_w = (Σ wᵢxᵢ) / (Σ wᵢ)"

#### concept_median
- **canonical_id**: `concept_median`
- **is_a**: `concept_central_tendency`
- **rpn_sketch**: `[RECALL sorted][RECALL n][GALAXY_LOOKUP star.constant.half][TMUL][TPACK 1][STORE median]`
- **symlinks**: `star.letter.n, star.symbol.equal`
- **surface_forms**:
  - en: "median = middle value of sorted data"
  - pt: "mediana = valor central dos dados ordenados"
  - es: "mediana = valor central de los datos ordenados"
  - fr: "médiane = valeur centrale des données triées"
  - de: "Median = mittlerer Wert der sortierten Daten"
  - it: "mediana = valore centrale dei dati ordinati"
  - ja: "中央値 = 並べ替えたデータの中央の値"
  - zh: "中位数 = 排序数据的中间值"
  - ru: "медиана = центральное значение упорядоченных данных"

#### concept_mode
- **canonical_id**: `concept_mode`
- **is_a**: `concept_central_tendency`
- **rpn_sketch**: `[RECALL data][GALAXY_LOOKUP star.symbol.argmax_count][TPACK 1][STORE mode]`
- **symlinks**: `star.symbol.equal`
- **surface_forms**:
  - en: "mode = most frequent value"
  - pt: "moda = valor mais frequente"
  - es: "moda = valor más frecuente"
  - fr: "mode = valeur la plus fréquente"
  - de: "Modus = häufigster Wert"
  - it: "moda = valore più frequente"
  - ja: "最頻値 = 最も頻繁に現れる値"
  - zh: "众数 = 出现次数最多的值"
  - ru: "мода = наиболее частое значение"

#### formula_range
- **canonical_id**: `formula_statistical_range`
- **is_a**: `formula_dispersion`
- **rpn_sketch**: `[RECALL max][RECALL min][TNOT][TADD][STORE range]`
- **symlinks**: `star.symbol.minus, star.symbol.equal`
- **surface_forms**:
  - en: "range = max − min"
  - pt: "amplitude = máximo − mínimo"
  - es: "rango = máximo − mínimo"
  - fr: "étendue = max − min"
  - de: "Spannweite = Maximum − Minimum"
  - it: "range = massimo − minimo"
  - ja: "範囲 = 最大値 − 最小値"
  - zh: "极差 = 最大值 − 最小值"
  - ru: "размах = максимум − минимум"

#### formula_variance_population
- **canonical_id**: `formula_variance_population`
- **is_a**: `formula_dispersion`
- **rpn_sketch**: `[RECALL x_i][GALAXY_LOOKUP star.letter.mu][TNOT][TADD][RECALL x_i][GALAXY_LOOKUP star.letter.mu][TNOT][TADD][TMUL][GALAXY_LOOKUP star.symbol.sum][TPACK 1][RECALL N][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE variance]`
- **symlinks**: `star.letter.x, star.letter.mu, star.letter.N, star.symbol.sum, star.symbol.equal`
- **surface_forms**:
  - en: "σ² = Σ(xᵢ − μ)² / N"
  - pt: "σ² = Σ(xᵢ − μ)² / N"
  - es: "σ² = Σ(xᵢ − μ)² / N"
  - fr: "σ² = Σ(xᵢ − μ)² / N"
  - de: "σ² = Σ(xᵢ − μ)² / N"
  - it: "σ² = Σ(xᵢ − μ)² / N"
  - ja: "σ² = Σ(xᵢ − μ)² / N"
  - zh: "σ² = Σ(xᵢ − μ)² / N"
  - ru: "σ² = Σ(xᵢ − μ)² / N"

#### formula_variance_sample
- **canonical_id**: `formula_variance_sample`
- **is_a**: `formula_dispersion`
- **rpn_sketch**: `[RECALL x_i][RECALL xbar][TNOT][TADD][RECALL x_i][RECALL xbar][TNOT][TADD][TMUL][GALAXY_LOOKUP star.symbol.sum][TPACK 1][RECALL n][GALAXY_LOOKUP star.constant.one][TNOT][TADD][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE variance]`
- **symlinks**: `star.letter.x, star.letter.n, star.symbol.sum, star.symbol.equal`
- **surface_forms**:
  - en: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - pt: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - es: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - fr: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - de: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - it: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - ja: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - zh: "s² = Σ(xᵢ − x̄)² / (n − 1)"
  - ru: "s² = Σ(xᵢ − x̄)² / (n − 1)"

#### formula_standard_deviation
- **canonical_id**: `formula_standard_deviation`
- **is_a**: `formula_dispersion`
- **rpn_sketch**: `[RECALL variance][GALAXY_LOOKUP star.symbol.sqrt][TPACK 1][STORE sigma]`
- **symlinks**: `star.letter.sigma, star.symbol.sqrt, star.symbol.equal`
- **surface_forms**:
  - en: "σ = √(σ²)"
  - pt: "σ = √(σ²)"
  - es: "σ = √(σ²)"
  - fr: "σ = √(σ²)"
  - de: "σ = √(σ²)"
  - it: "σ = √(σ²)"
  - ja: "σ = √(σ²)"
  - zh: "σ = √(σ²)"
  - ru: "σ = √(σ²)"

#### formula_z_score
- **canonical_id**: `formula_z_score`
- **is_a**: `formula_normalisation`
- **rpn_sketch**: `[RECALL x][GALAXY_LOOKUP star.letter.mu][TNOT][TADD][GALAXY_LOOKUP star.letter.sigma][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE z]`
- **symlinks**: `star.letter.x, star.letter.mu, star.letter.sigma, star.letter.z, star.symbol.equal`
- **surface_forms**:
  - en: "z = (x − μ) / σ"
  - pt: "z = (x − μ) / σ"
  - es: "z = (x − μ) / σ"
  - fr: "z = (x − μ) / σ"
  - de: "z = (x − μ) / σ"
  - it: "z = (x − μ) / σ"
  - ja: "z = (x − μ) / σ"
  - zh: "z = (x − μ) / σ"
  - ru: "z = (x − μ) / σ"

#### formula_correlation_coefficient
- **canonical_id**: `formula_correlation_coefficient_pearson`
- **is_a**: `formula_correlation`
- **rpn_sketch**: `[RECALL x_i][RECALL xbar][TNOT][TADD][RECALL y_i][RECALL ybar][TNOT][TADD][TMUL][GALAXY_LOOKUP star.symbol.sum][TPACK 1][RECALL Sxx][RECALL Syy][TMUL][GALAXY_LOOKUP star.symbol.sqrt][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE r]`
- **symlinks**: `star.letter.r, star.letter.x, star.letter.y, star.symbol.sum, star.symbol.equal`
- **surface_forms**:
  - en: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - pt: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - es: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - fr: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - de: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - it: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - ja: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - zh: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"
  - ru: "r = Σ(xᵢ − x̄)(yᵢ − ȳ) / √(Sₓₓ · S_yy)"

#### formula_least_squares_slope
- **canonical_id**: `formula_least_squares_slope`
- **is_a**: `formula_regression`
- **rpn_sketch**: `[RECALL r][RECALL sy][TMUL][RECALL sx][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE b]`
- **symlinks**: `star.letter.r, star.letter.x, star.letter.y, star.letter.b, star.symbol.equal`
- **surface_forms**:
  - en: "b = r · (s_y / s_x)"
  - pt: "b = r · (s_y / s_x)"
  - es: "b = r · (s_y / s_x)"
  - fr: "b = r · (s_y / s_x)"
  - de: "b = r · (s_y / s_x)"
  - it: "b = r · (s_y / s_x)"
  - ja: "b = r · (s_y / s_x)"
  - zh: "b = r · (s_y / s_x)"
  - ru: "b = r · (s_y / s_x)"

## Cluster 3.B — Probability

#### formula_probability_classical
- **canonical_id**: `formula_probability_classical`
- **is_a**: `formula_probability_axiom`
- **rpn_sketch**: `[RECALL favorable][RECALL total][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE P]`
- **symlinks**: `star.letter.P, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "P(A) = favorable outcomes / total outcomes"
  - pt: "P(A) = casos favoráveis / casos totais"
  - es: "P(A) = casos favorables / casos totales"
  - fr: "P(A) = cas favorables / cas possibles"
  - de: "P(A) = günstige Fälle / mögliche Fälle"
  - it: "P(A) = casi favorevoli / casi totali"
  - ja: "P(A) = 有利な場合の数 / 全体の場合の数"
  - zh: "P(A) = 有利结果数 / 总结果数"
  - ru: "P(A) = благоприятные исходы / все исходы"

#### formula_probability_complement
- **canonical_id**: `formula_probability_complement`
- **is_a**: `formula_probability_axiom`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.one][RECALL P_A][TNOT][TADD][STORE P_not_A]`
- **symlinks**: `star.letter.P, star.symbol.minus, star.symbol.equal`
- **surface_forms**:
  - en: "P(A') = 1 − P(A)"
  - pt: "P(A') = 1 − P(A)"
  - es: "P(A') = 1 − P(A)"
  - fr: "P(A') = 1 − P(A)"
  - de: "P(A') = 1 − P(A)"
  - it: "P(A') = 1 − P(A)"
  - ja: "P(A') = 1 − P(A)"
  - zh: "P(A') = 1 − P(A)"
  - ru: "P(A') = 1 − P(A)"

#### formula_probability_union
- **canonical_id**: `formula_probability_union`
- **is_a**: `formula_probability_axiom`
- **rpn_sketch**: `[RECALL P_A][RECALL P_B][TADD][RECALL P_AB][TNOT][TADD][STORE P_union]`
- **symlinks**: `star.letter.P, star.symbol.plus, star.symbol.minus, star.symbol.equal`
- **surface_forms**:
  - en: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - pt: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - es: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - fr: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - de: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - it: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - ja: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - zh: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"
  - ru: "P(A ∪ B) = P(A) + P(B) − P(A ∩ B)"

#### formula_probability_intersection_independent
- **canonical_id**: `formula_probability_intersection_independent`
- **is_a**: `formula_probability_axiom`
- **rpn_sketch**: `[RECALL P_A][RECALL P_B][TMUL][STORE P_AB]`
- **symlinks**: `star.letter.P, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "P(A ∩ B) = P(A) · P(B) (independent)"
  - pt: "P(A ∩ B) = P(A) · P(B) (independentes)"
  - es: "P(A ∩ B) = P(A) · P(B) (independientes)"
  - fr: "P(A ∩ B) = P(A) · P(B) (indépendants)"
  - de: "P(A ∩ B) = P(A) · P(B) (unabhängig)"
  - it: "P(A ∩ B) = P(A) · P(B) (indipendenti)"
  - ja: "P(A ∩ B) = P(A) · P(B)（独立）"
  - zh: "P(A ∩ B) = P(A) · P(B)（独立）"
  - ru: "P(A ∩ B) = P(A) · P(B) (независимые)"

#### formula_conditional_probability
- **canonical_id**: `formula_conditional_probability`
- **is_a**: `formula_probability_definition`
- **rpn_sketch**: `[RECALL P_AB][RECALL P_B][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE P_A_given_B]`
- **symlinks**: `star.letter.P, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "P(A | B) = P(A ∩ B) / P(B)"
  - pt: "P(A | B) = P(A ∩ B) / P(B)"
  - es: "P(A | B) = P(A ∩ B) / P(B)"
  - fr: "P(A | B) = P(A ∩ B) / P(B)"
  - de: "P(A | B) = P(A ∩ B) / P(B)"
  - it: "P(A | B) = P(A ∩ B) / P(B)"
  - ja: "P(A | B) = P(A ∩ B) / P(B)"
  - zh: "P(A | B) = P(A ∩ B) / P(B)"
  - ru: "P(A | B) = P(A ∩ B) / P(B)"

#### theorem_bayes
- **canonical_id**: `theorem_bayes`
- **is_a**: `theorem_probability`
- **rpn_sketch**: `[RECALL P_B_given_A][RECALL P_A][TMUL][RECALL P_B][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE P_A_given_B]`
- **symlinks**: `star.letter.P, star.symbol.divide, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "P(A | B) = P(B | A) · P(A) / P(B)"
  - pt: "P(A | B) = P(B | A) · P(A) / P(B)"
  - es: "P(A | B) = P(B | A) · P(A) / P(B)"
  - fr: "P(A | B) = P(B | A) · P(A) / P(B)"
  - de: "P(A | B) = P(B | A) · P(A) / P(B)"
  - it: "P(A | B) = P(B | A) · P(A) / P(B)"
  - ja: "P(A | B) = P(B | A) · P(A) / P(B)"
  - zh: "P(A | B) = P(B | A) · P(A) / P(B)"
  - ru: "P(A | B) = P(B | A) · P(A) / P(B)"

#### formula_expected_value_discrete
- **canonical_id**: `formula_expected_value_discrete`
- **is_a**: `formula_random_variable`
- **rpn_sketch**: `[RECALL x_i][RECALL P_i][TMUL][GALAXY_LOOKUP star.symbol.sum][TPACK 1][STORE E]`
- **symlinks**: `star.letter.x, star.letter.P, star.symbol.sum, star.symbol.equal`
- **surface_forms**:
  - en: "E(X) = Σ xᵢ · P(xᵢ)"
  - pt: "E(X) = Σ xᵢ · P(xᵢ)"
  - es: "E(X) = Σ xᵢ · P(xᵢ)"
  - fr: "E(X) = Σ xᵢ · P(xᵢ)"
  - de: "E(X) = Σ xᵢ · P(xᵢ)"
  - it: "E(X) = Σ xᵢ · P(xᵢ)"
  - ja: "E(X) = Σ xᵢ · P(xᵢ)"
  - zh: "E(X) = Σ xᵢ · P(xᵢ)"
  - ru: "E(X) = Σ xᵢ · P(xᵢ)"

#### formula_binomial_distribution
- **canonical_id**: `formula_binomial_distribution`
- **is_a**: `formula_discrete_distribution`
- **rpn_sketch**: `[RECALL n][RECALL k][GALAXY_LOOKUP star.symbol.binom][TPACK 2][RECALL p][RECALL k][GALAXY_LOOKUP star.symbol.power][TPACK 2][TMUL][GALAXY_LOOKUP star.constant.one][RECALL p][TNOT][TADD][RECALL n][RECALL k][TNOT][TADD][GALAXY_LOOKUP star.symbol.power][TPACK 2][TMUL][STORE P_X]`
- **symlinks**: `star.letter.n, star.letter.k, star.letter.p, star.symbol.power, star.symbol.equal`
- **surface_forms**:
  - en: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - pt: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - es: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - fr: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - de: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - it: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - ja: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - zh: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"
  - ru: "P(X = k) = C(n, k) · p^k · (1−p)^(n−k)"

#### formula_binomial_mean
- **canonical_id**: `formula_binomial_mean`
- **is_a**: `formula_discrete_distribution`
- **rpn_sketch**: `[RECALL n][RECALL p][TMUL][STORE mu]`
- **symlinks**: `star.letter.n, star.letter.p, star.letter.mu, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "μ = n · p"
  - pt: "μ = n · p"
  - es: "μ = n · p"
  - fr: "μ = n · p"
  - de: "μ = n · p"
  - it: "μ = n · p"
  - ja: "μ = n · p"
  - zh: "μ = n · p"
  - ru: "μ = n · p"

#### formula_binomial_variance
- **canonical_id**: `formula_binomial_variance`
- **is_a**: `formula_discrete_distribution`
- **rpn_sketch**: `[RECALL n][RECALL p][TMUL][GALAXY_LOOKUP star.constant.one][RECALL p][TNOT][TADD][TMUL][STORE sigma_squared]`
- **symlinks**: `star.letter.n, star.letter.p, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "σ² = n · p · (1 − p)"
  - pt: "σ² = n · p · (1 − p)"
  - es: "σ² = n · p · (1 − p)"
  - fr: "σ² = n · p · (1 − p)"
  - de: "σ² = n · p · (1 − p)"
  - it: "σ² = n · p · (1 − p)"
  - ja: "σ² = n · p · (1 − p)"
  - zh: "σ² = n · p · (1 − p)"
  - ru: "σ² = n · p · (1 − p)"

#### rule_empirical_normal
- **canonical_id**: `rule_empirical_normal_68_95_99`
- **is_a**: `rule_normal_distribution`
- **rpn_sketch**: `[RECALL sigma_count][GALAXY_LOOKUP star.constant.one][TCOMP EQ][OP_BRANCH p_68][GALAXY_LOOKUP star.constant.two][TCOMP EQ][OP_BRANCH p_95][GALAXY_LOOKUP star.constant.three][TCOMP EQ][OP_BRANCH p_997]`
- **symlinks**: `star.letter.sigma, star.symbol.equal`
- **surface_forms**:
  - en: "≈68% within 1σ, ≈95% within 2σ, ≈99.7% within 3σ"
  - pt: "≈68% em 1σ, ≈95% em 2σ, ≈99,7% em 3σ"
  - es: "≈68% en 1σ, ≈95% en 2σ, ≈99,7% en 3σ"
  - fr: "≈68% à 1σ, ≈95% à 2σ, ≈99,7% à 3σ"
  - de: "≈68% innerhalb 1σ, ≈95% innerhalb 2σ, ≈99,7% innerhalb 3σ"
  - it: "≈68% in 1σ, ≈95% in 2σ, ≈99,7% in 3σ"
  - ja: "1σ で約68%、2σ で約95%、3σ で約99.7%"
  - zh: "1σ 约68%，2σ 约95%，3σ 约99.7%"
  - ru: "≈68% в 1σ, ≈95% в 2σ, ≈99,7% в 3σ"

## Cluster 3.C — Combinatorics

#### formula_factorial_definition
- **canonical_id**: `formula_factorial_definition`
- **is_a**: `formula_combinatorial_function`
- **rpn_sketch**: `[RECALL n][RECALL n][GALAXY_LOOKUP star.constant.one][TNOT][TADD][TMUL][RECALL n][GALAXY_LOOKUP star.constant.two][TNOT][TADD][TMUL][TPACK n][STORE n_factorial]`
- **symlinks**: `star.letter.n, star.symbol.equal`
- **surface_forms**:
  - en: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - pt: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - es: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - fr: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - de: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - it: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - ja: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - zh: "n! = n · (n−1) · (n−2) · ⋯ · 1"
  - ru: "n! = n · (n−1) · (n−2) · ⋯ · 1"

#### formula_permutation_n_r
- **canonical_id**: `formula_permutation_n_r`
- **is_a**: `formula_counting`
- **rpn_sketch**: `[RECALL n][GALAXY_LOOKUP star.symbol.factorial][TPACK 1][RECALL n][RECALL r][TNOT][TADD][GALAXY_LOOKUP star.symbol.factorial][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE P_nr]`
- **symlinks**: `star.letter.n, star.letter.r, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "P(n, r) = n! / (n − r)!"
  - pt: "P(n, r) = n! / (n − r)!"
  - es: "P(n, r) = n! / (n − r)!"
  - fr: "P(n, r) = n! / (n − r)!"
  - de: "P(n, r) = n! / (n − r)!"
  - it: "P(n, r) = n! / (n − r)!"
  - ja: "P(n, r) = n! / (n − r)!"
  - zh: "P(n, r) = n! / (n − r)!"
  - ru: "P(n, r) = n! / (n − r)!"

#### formula_combination_n_r
- **canonical_id**: `formula_combination_n_r`
- **is_a**: `formula_counting`
- **rpn_sketch**: `[RECALL n][GALAXY_LOOKUP star.symbol.factorial][TPACK 1][RECALL r][GALAXY_LOOKUP star.symbol.factorial][TPACK 1][RECALL n][RECALL r][TNOT][TADD][GALAXY_LOOKUP star.symbol.factorial][TPACK 1][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE C_nr]`
- **symlinks**: `star.letter.n, star.letter.r, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "C(n, r) = n! / (r! · (n−r)!)"
  - pt: "C(n, r) = n! / (r! · (n−r)!)"
  - es: "C(n, r) = n! / (r! · (n−r)!)"
  - fr: "C(n, r) = n! / (r! · (n−r)!)"
  - de: "C(n, r) = n! / (r! · (n−r)!)"
  - it: "C(n, r) = n! / (r! · (n−r)!)"
  - ja: "C(n, r) = n! / (r! · (n−r)!)"
  - zh: "C(n, r) = n! / (r! · (n−r)!)"
  - ru: "C(n, r) = n! / (r! · (n−r)!)"

#### identity_pascal_rule
- **canonical_id**: `identity_pascal_rule`
- **is_a**: `identity_combinatorial`
- **rpn_sketch**: `[RECALL n][GALAXY_LOOKUP star.constant.one][TNOT][TADD][RECALL k][GALAXY_LOOKUP star.symbol.binom][TPACK 2][RECALL n][GALAXY_LOOKUP star.constant.one][TNOT][TADD][RECALL k][GALAXY_LOOKUP star.constant.one][TNOT][TADD][GALAXY_LOOKUP star.symbol.binom][TPACK 2][TADD][RECALL n][RECALL k][GALAXY_LOOKUP star.symbol.binom][TPACK 2][TCOMP EQ]`
- **symlinks**: `star.letter.n, star.letter.k, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - pt: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - es: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - fr: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - de: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - it: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - ja: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - zh: "C(n−1, k) + C(n−1, k−1) = C(n, k)"
  - ru: "C(n−1, k) + C(n−1, k−1) = C(n, k)"

#### concept_pigeonhole
- **canonical_id**: `concept_pigeonhole`
- **is_a**: `concept_combinatorial_principle`
- **rpn_sketch**: `[RECALL items][RECALL boxes][TCOMP GT][OP_BRANCH some_box_has_two]`
- **symlinks**: `star.symbol.greater, star.symbol.equal`
- **surface_forms**:
  - en: "if n+1 items go into n boxes, some box has ≥2 items"
  - pt: "se n+1 objetos vão em n caixas, alguma caixa tem ≥2 objetos"
  - es: "si n+1 objetos van en n cajas, alguna caja tiene ≥2 objetos"
  - fr: "si n+1 objets vont dans n boîtes, une boîte contient ≥2 objets"
  - de: "wenn n+1 Objekte in n Schubfächer kommen, enthält eines ≥2 Objekte"
  - it: "se n+1 oggetti vanno in n scatole, una scatola ha ≥2 oggetti"
  - ja: "n+1 個を n 箱に入れると、ある箱には 2 個以上ある"
  - zh: "若 n+1 个物体放入 n 个盒子，则某盒至少有 2 个"
  - ru: "если n+1 объект помещается в n ящиков, в каком-то ящике ≥2 объектов"

## Cluster 3.D — Discrete Math / Sets / Logic

#### formula_set_cardinality_union
- **canonical_id**: `formula_set_cardinality_union`
- **is_a**: `formula_set_theoretic`
- **rpn_sketch**: `[RECALL card_A][RECALL card_B][TADD][RECALL card_AB][TNOT][TADD][STORE card_union]`
- **symlinks**: `star.letter.A, star.letter.B, star.symbol.plus, star.symbol.minus, star.symbol.equal`
- **surface_forms**:
  - en: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - pt: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - es: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - fr: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - de: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - it: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - ja: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - zh: "|A ∪ B| = |A| + |B| − |A ∩ B|"
  - ru: "|A ∪ B| = |A| + |B| − |A ∩ B|"

#### identity_de_morgan_sets
- **canonical_id**: `identity_de_morgan_sets`
- **is_a**: `identity_set_theoretic`
- **rpn_sketch**: `[RECALL A][RECALL B][GALAXY_LOOKUP star.symbol.union][TPACK 2][GALAXY_LOOKUP star.symbol.complement][TPACK 1][RECALL A][GALAXY_LOOKUP star.symbol.complement][TPACK 1][RECALL B][GALAXY_LOOKUP star.symbol.complement][TPACK 1][GALAXY_LOOKUP star.symbol.intersect][TPACK 2][TCOMP EQ]`
- **symlinks**: `star.letter.A, star.letter.B, star.symbol.equal`
- **surface_forms**:
  - en: "(A ∪ B)' = A' ∩ B'"
  - pt: "(A ∪ B)' = A' ∩ B'"
  - es: "(A ∪ B)' = A' ∩ B'"
  - fr: "(A ∪ B)' = A' ∩ B'"
  - de: "(A ∪ B)' = A' ∩ B'"
  - it: "(A ∪ B)' = A' ∩ B'"
  - ja: "(A ∪ B)' = A' ∩ B'"
  - zh: "(A ∪ B)' = A' ∩ B'"
  - ru: "(A ∪ B)' = A' ∩ B'"

#### formula_power_set_cardinality
- **canonical_id**: `formula_power_set_cardinality`
- **is_a**: `formula_set_theoretic`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.two][RECALL n][GALAXY_LOOKUP star.symbol.power][TPACK 2][STORE card_power_set]`
- **symlinks**: `star.letter.n, star.symbol.power, star.symbol.equal`
- **surface_forms**:
  - en: "|P(A)| = 2^n  (where n = |A|)"
  - pt: "|P(A)| = 2^n  (onde n = |A|)"
  - es: "|P(A)| = 2^n  (donde n = |A|)"
  - fr: "|P(A)| = 2^n  (où n = |A|)"
  - de: "|P(A)| = 2^n  (mit n = |A|)"
  - it: "|P(A)| = 2^n  (dove n = |A|)"
  - ja: "|P(A)| = 2^n（n = |A|）"
  - zh: "|P(A)| = 2^n（n = |A|）"
  - ru: "|P(A)| = 2^n  (где n = |A|)"

#### identity_de_morgan_logic
- **canonical_id**: `identity_de_morgan_logic`
- **is_a**: `identity_propositional_logic`
- **rpn_sketch**: `[RECALL p][RECALL q][TPACK 2][TNOT][RECALL p][TNOT][RECALL q][TNOT][TPACK 2][TCOMP EQ]`
- **symlinks**: `star.letter.p, star.letter.q, star.symbol.equal`
- **surface_forms**:
  - en: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - pt: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - es: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - fr: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - de: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - it: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - ja: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - zh: "¬(p ∧ q) ≡ ¬p ∨ ¬q"
  - ru: "¬(p ∧ q) ≡ ¬p ∨ ¬q"

#### identity_contrapositive
- **canonical_id**: `identity_contrapositive`
- **is_a**: `identity_propositional_logic`
- **rpn_sketch**: `[RECALL p][RECALL q][GALAXY_LOOKUP star.symbol.implies][TPACK 2][RECALL q][TNOT][RECALL p][TNOT][GALAXY_LOOKUP star.symbol.implies][TPACK 2][TCOMP EQ]`
- **symlinks**: `star.letter.p, star.letter.q, star.symbol.equal`
- **surface_forms**:
  - en: "(p → q) ≡ (¬q → ¬p)"
  - pt: "(p → q) ≡ (¬q → ¬p)"
  - es: "(p → q) ≡ (¬q → ¬p)"
  - fr: "(p → q) ≡ (¬q → ¬p)"
  - de: "(p → q) ≡ (¬q → ¬p)"
  - it: "(p → q) ≡ (¬q → ¬p)"
  - ja: "(p → q) ≡ (¬q → ¬p)"
  - zh: "(p → q) ≡ (¬q → ¬p)"
  - ru: "(p → q) ≡ (¬q → ¬p)"

#### theorem_handshake
- **canonical_id**: `theorem_handshake_graph`
- **is_a**: `theorem_graph_theory`
- **rpn_sketch**: `[GALAXY_LOOKUP star.symbol.sum][TPACK 1][GALAXY_LOOKUP star.constant.two][RECALL E][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.E, star.symbol.sum, star.symbol.equal`
- **surface_forms**:
  - en: "Σ deg(v) = 2 · |E|"
  - pt: "Σ deg(v) = 2 · |E|"
  - es: "Σ deg(v) = 2 · |E|"
  - fr: "Σ deg(v) = 2 · |E|"
  - de: "Σ deg(v) = 2 · |E|"
  - it: "Σ deg(v) = 2 · |E|"
  - ja: "Σ deg(v) = 2 · |E|"
  - zh: "Σ deg(v) = 2 · |E|"
  - ru: "Σ deg(v) = 2 · |E|"

## Cluster 3.E — Financial Math

#### formula_simple_interest
- **canonical_id**: `formula_simple_interest`
- **is_a**: `formula_financial`
- **rpn_sketch**: `[RECALL P][RECALL r][TMUL][RECALL t][TMUL][STORE I]`
- **symlinks**: `star.letter.P, star.letter.r, star.letter.t, star.letter.I, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "I = P · r · t"
  - pt: "J = C · i · t"
  - es: "I = C · r · t"
  - fr: "I = C · r · t"
  - de: "Z = K · p · t"
  - it: "I = C · r · t"
  - ja: "利息 = 元金 · 利率 · 期間"
  - zh: "利息 = 本金 · 利率 · 时间"
  - ru: "I = P · r · t"
- **saudades**: `true`

#### formula_compound_interest_periodic
- **canonical_id**: `formula_compound_interest_periodic`
- **is_a**: `formula_financial`
- **rpn_sketch**: `[RECALL P][GALAXY_LOOKUP star.constant.one][RECALL r][RECALL n][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TADD][RECALL n][RECALL t][TMUL][GALAXY_LOOKUP star.symbol.power][TPACK 2][TMUL][STORE A]`
- **symlinks**: `star.letter.P, star.letter.r, star.letter.n, star.letter.t, star.symbol.power, star.symbol.equal`
- **surface_forms**:
  - en: "A = P · (1 + r/n)^(n·t)"
  - pt: "M = C · (1 + i/n)^(n·t)"
  - es: "A = C · (1 + r/n)^(n·t)"
  - fr: "A = C · (1 + r/n)^(n·t)"
  - de: "A = K · (1 + r/n)^(n·t)"
  - it: "A = C · (1 + r/n)^(n·t)"
  - ja: "A = P · (1 + r/n)^(n·t)"
  - zh: "A = P · (1 + r/n)^(n·t)"
  - ru: "A = P · (1 + r/n)^(n·t)"
- **saudades**: `true`

#### formula_compound_interest_continuous
- **canonical_id**: `formula_compound_interest_continuous`
- **is_a**: `formula_financial`
- **rpn_sketch**: `[RECALL P][GALAXY_LOOKUP star.constant.e][RECALL r][RECALL t][TMUL][GALAXY_LOOKUP star.symbol.power][TPACK 2][TMUL][STORE A]`
- **symlinks**: `star.letter.P, star.letter.r, star.letter.t, star.constant.e, star.symbol.equal`
- **surface_forms**:
  - en: "A = P · e^(r · t)"
  - pt: "M = C · e^(i · t)"
  - es: "A = C · e^(r · t)"
  - fr: "A = C · e^(r · t)"
  - de: "A = K · e^(r · t)"
  - it: "A = C · e^(r · t)"
  - ja: "A = P · e^(r · t)"
  - zh: "A = P · e^(r · t)"
  - ru: "A = P · e^(r · t)"

#### formula_present_value
- **canonical_id**: `formula_present_value`
- **is_a**: `formula_financial`
- **rpn_sketch**: `[RECALL FV][GALAXY_LOOKUP star.constant.one][RECALL r][TADD][RECALL t][GALAXY_LOOKUP star.symbol.power][TPACK 2][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE PV]`
- **symlinks**: `star.letter.r, star.letter.t, star.symbol.power, star.symbol.equal`
- **surface_forms**:
  - en: "PV = FV / (1 + r)^t"
  - pt: "VP = VF / (1 + i)^t"
  - es: "VP = VF / (1 + r)^t"
  - fr: "VA = VF / (1 + r)^t"
  - de: "BW = EW / (1 + r)^t"
  - it: "VA = VF / (1 + r)^t"
  - ja: "現在価値 = 将来価値 / (1 + r)^t"
  - zh: "现值 = 终值 / (1 + r)^t"
  - ru: "PV = FV / (1 + r)^t"

#### formula_percentage_increase_amount
- **canonical_id**: `formula_percentage_increase_amount`
- **is_a**: `formula_percent_change`
- **rpn_sketch**: `[RECALL old][RECALL r][TMUL][RECALL old][TADD][STORE new]`
- **symlinks**: `star.letter.r, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "new = old · (1 + r)"
  - pt: "novo = antigo · (1 + r)"
  - es: "nuevo = viejo · (1 + r)"
  - fr: "nouveau = ancien · (1 + r)"
  - de: "neu = alt · (1 + r)"
  - it: "nuovo = vecchio · (1 + r)"
  - ja: "新 = 旧 · (1 + r)"
  - zh: "新值 = 旧值 · (1 + r)"
  - ru: "новое = старое · (1 + r)"

## Cluster 3.F — Applied Physics (HS)

#### formula_kinematic_velocity_time
- **canonical_id**: `formula_kinematic_velocity_time`
- **is_a**: `formula_kinematics`
- **rpn_sketch**: `[RECALL u][RECALL a][RECALL t][TMUL][TADD][STORE v]`
- **symlinks**: `star.letter.u, star.letter.v, star.letter.a, star.letter.t, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "v = u + a · t"
  - pt: "v = u + a · t"
  - es: "v = u + a · t"
  - fr: "v = u + a · t"
  - de: "v = u + a · t"
  - it: "v = u + a · t"
  - ja: "v = u + a · t"
  - zh: "v = u + a · t"
  - ru: "v = u + a · t"

#### formula_kinematic_displacement
- **canonical_id**: `formula_kinematic_displacement`
- **is_a**: `formula_kinematics`
- **rpn_sketch**: `[RECALL u][RECALL t][TMUL][GALAXY_LOOKUP star.constant.half][RECALL a][TMUL][RECALL t][RECALL t][TMUL][TMUL][TADD][STORE s]`
- **symlinks**: `star.letter.u, star.letter.a, star.letter.t, star.letter.s, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "s = u · t + ½ · a · t²"
  - pt: "s = u · t + ½ · a · t²"
  - es: "s = u · t + ½ · a · t²"
  - fr: "s = u · t + ½ · a · t²"
  - de: "s = u · t + ½ · a · t²"
  - it: "s = u · t + ½ · a · t²"
  - ja: "s = u · t + ½ · a · t²"
  - zh: "s = u · t + ½ · a · t²"
  - ru: "s = u · t + ½ · a · t²"

#### formula_kinematic_velocity_squared
- **canonical_id**: `formula_kinematic_velocity_squared`
- **is_a**: `formula_kinematics`
- **rpn_sketch**: `[RECALL u][RECALL u][TMUL][GALAXY_LOOKUP star.constant.two][RECALL a][TMUL][RECALL s][TMUL][TADD][STORE v_squared]`
- **symlinks**: `star.letter.u, star.letter.v, star.letter.a, star.letter.s, star.symbol.equal`
- **surface_forms**:
  - en: "v² = u² + 2 · a · s"
  - pt: "v² = u² + 2 · a · s"
  - es: "v² = u² + 2 · a · s"
  - fr: "v² = u² + 2 · a · s"
  - de: "v² = u² + 2 · a · s"
  - it: "v² = u² + 2 · a · s"
  - ja: "v² = u² + 2 · a · s"
  - zh: "v² = u² + 2 · a · s"
  - ru: "v² = u² + 2 · a · s"

#### formula_newton_second_law
- **canonical_id**: `formula_newton_second_law`
- **is_a**: `formula_dynamics`
- **rpn_sketch**: `[RECALL m][RECALL a][TMUL][STORE F]`
- **symlinks**: `star.letter.F, star.letter.m, star.letter.a, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "F = m · a"
  - pt: "F = m · a"
  - es: "F = m · a"
  - fr: "F = m · a"
  - de: "F = m · a"
  - it: "F = m · a"
  - ja: "F = m · a"
  - zh: "F = m · a"
  - ru: "F = m · a"

#### formula_kinetic_energy
- **canonical_id**: `formula_kinetic_energy`
- **is_a**: `formula_energy`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.half][RECALL m][TMUL][RECALL v][RECALL v][TMUL][TMUL][STORE KE]`
- **symlinks**: `star.letter.m, star.letter.v, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "KE = ½ · m · v²"
  - pt: "Ec = ½ · m · v²"
  - es: "Ec = ½ · m · v²"
  - fr: "Ec = ½ · m · v²"
  - de: "E_kin = ½ · m · v²"
  - it: "Ec = ½ · m · v²"
  - ja: "運動エネルギー = ½ · m · v²"
  - zh: "动能 = ½ · m · v²"
  - ru: "Eₖ = ½ · m · v²"

#### formula_gravitational_potential_energy
- **canonical_id**: `formula_gravitational_potential_energy`
- **is_a**: `formula_energy`
- **rpn_sketch**: `[RECALL m][RECALL g][TMUL][RECALL h][TMUL][STORE PE]`
- **symlinks**: `star.letter.m, star.letter.g, star.letter.h, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "PE = m · g · h"
  - pt: "Ep = m · g · h"
  - es: "Ep = m · g · h"
  - fr: "Ep = m · g · h"
  - de: "E_pot = m · g · h"
  - it: "Ep = m · g · h"
  - ja: "位置エネルギー = m · g · h"
  - zh: "重力势能 = m · g · h"
  - ru: "Eₚ = m · g · h"

#### formula_momentum
- **canonical_id**: `formula_linear_momentum`
- **is_a**: `formula_dynamics`
- **rpn_sketch**: `[RECALL m][RECALL v][TMUL][STORE p]`
- **symlinks**: `star.letter.m, star.letter.v, star.letter.p, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "p = m · v"
  - pt: "p = m · v"
  - es: "p = m · v"
  - fr: "p = m · v"
  - de: "p = m · v"
  - it: "p = m · v"
  - ja: "p = m · v"
  - zh: "p = m · v"
  - ru: "p = m · v"

#### formula_universal_gravitation
- **canonical_id**: `formula_universal_gravitation`
- **is_a**: `formula_dynamics`
- **rpn_sketch**: `[RECALL G][RECALL m1][RECALL m2][TMUL][TMUL][RECALL r][RECALL r][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE F]`
- **symlinks**: `star.letter.G, star.letter.m, star.letter.r, star.letter.F, star.symbol.equal`
- **surface_forms**:
  - en: "F = G · m₁ · m₂ / r²"
  - pt: "F = G · m₁ · m₂ / r²"
  - es: "F = G · m₁ · m₂ / r²"
  - fr: "F = G · m₁ · m₂ / r²"
  - de: "F = G · m₁ · m₂ / r²"
  - it: "F = G · m₁ · m₂ / r²"
  - ja: "F = G · m₁ · m₂ / r²"
  - zh: "F = G · m₁ · m₂ / r²"
  - ru: "F = G · m₁ · m₂ / r²"

#### formula_ohms_law
- **canonical_id**: `formula_ohms_law`
- **is_a**: `formula_electromagnetism`
- **rpn_sketch**: `[RECALL I][RECALL R][TMUL][STORE V]`
- **symlinks**: `star.letter.V, star.letter.I, star.letter.R, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "V = I · R"
  - pt: "V = I · R"
  - es: "V = I · R"
  - fr: "V = I · R"
  - de: "U = I · R"
  - it: "V = I · R"
  - ja: "V = I · R"
  - zh: "V = I · R"
  - ru: "U = I · R"

#### formula_wave_speed
- **canonical_id**: `formula_wave_speed`
- **is_a**: `formula_waves`
- **rpn_sketch**: `[RECALL f][GALAXY_LOOKUP star.letter.lambda][TMUL][STORE v]`
- **symlinks**: `star.letter.f, star.letter.lambda, star.letter.v, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "v = f · λ"
  - pt: "v = f · λ"
  - es: "v = f · λ"
  - fr: "v = f · λ"
  - de: "v = f · λ"
  - it: "v = f · λ"
  - ja: "v = f · λ"
  - zh: "v = f · λ"
  - ru: "v = f · λ"

#### formula_density
- **canonical_id**: `formula_density`
- **is_a**: `formula_continuum_mechanics`
- **rpn_sketch**: `[RECALL m][RECALL V][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE rho]`
- **symlinks**: `star.letter.m, star.letter.V, star.letter.rho, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "ρ = m / V"
  - pt: "ρ = m / V"
  - es: "ρ = m / V"
  - fr: "ρ = m / V"
  - de: "ρ = m / V"
  - it: "ρ = m / V"
  - ja: "ρ = m / V"
  - zh: "ρ = m / V"
  - ru: "ρ = m / V"

#### formula_pressure
- **canonical_id**: `formula_pressure`
- **is_a**: `formula_continuum_mechanics`
- **rpn_sketch**: `[RECALL F][RECALL A][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE P]`
- **symlinks**: `star.letter.F, star.letter.A, star.letter.P, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "P = F / A"
  - pt: "P = F / A"
  - es: "P = F / A"
  - fr: "P = F / A"
  - de: "p = F / A"
  - it: "P = F / A"
  - ja: "P = F / A"
  - zh: "P = F / A"
  - ru: "P = F / A"

#### formula_ideal_gas
- **canonical_id**: `formula_ideal_gas`
- **is_a**: `formula_thermodynamics`
- **rpn_sketch**: `[RECALL n][RECALL R][TMUL][RECALL T][TMUL][RECALL V][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE P]`
- **symlinks**: `star.letter.P, star.letter.V, star.letter.n, star.letter.R, star.letter.T, star.symbol.equal`
- **surface_forms**:
  - en: "P · V = n · R · T"
  - pt: "P · V = n · R · T"
  - es: "P · V = n · R · T"
  - fr: "P · V = n · R · T"
  - de: "P · V = n · R · T"
  - it: "P · V = n · R · T"
  - ja: "P · V = n · R · T"
  - zh: "P · V = n · R · T"
  - ru: "P · V = n · R · T"

## Cluster 3.G — Chemistry Math

#### formula_mole_from_mass
- **canonical_id**: `formula_mole_from_mass`
- **is_a**: `formula_chemistry`
- **rpn_sketch**: `[RECALL m][RECALL M][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE n]`
- **symlinks**: `star.letter.m, star.letter.M, star.letter.n, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "n = m / M"
  - pt: "n = m / M"
  - es: "n = m / M"
  - fr: "n = m / M"
  - de: "n = m / M"
  - it: "n = m / M"
  - ja: "n = m / M"
  - zh: "n = m / M"
  - ru: "n = m / M"

#### formula_molarity
- **canonical_id**: `formula_molarity`
- **is_a**: `formula_chemistry`
- **rpn_sketch**: `[RECALL n][RECALL V][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE c]`
- **symlinks**: `star.letter.n, star.letter.V, star.letter.c, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "c = n / V"
  - pt: "c = n / V"
  - es: "c = n / V"
  - fr: "c = n / V"
  - de: "c = n / V"
  - it: "c = n / V"
  - ja: "c = n / V"
  - zh: "c = n / V"
  - ru: "c = n / V"

#### formula_dilution
- **canonical_id**: `formula_dilution`
- **is_a**: `formula_chemistry`
- **rpn_sketch**: `[RECALL M1][RECALL V1][TMUL][RECALL M2][RECALL V2][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.M, star.letter.V, star.symbol.equal`
- **surface_forms**:
  - en: "M₁ · V₁ = M₂ · V₂"
  - pt: "M₁ · V₁ = M₂ · V₂"
  - es: "M₁ · V₁ = M₂ · V₂"
  - fr: "M₁ · V₁ = M₂ · V₂"
  - de: "M₁ · V₁ = M₂ · V₂"
  - it: "M₁ · V₁ = M₂ · V₂"
  - ja: "M₁ · V₁ = M₂ · V₂"
  - zh: "M₁ · V₁ = M₂ · V₂"
  - ru: "M₁ · V₁ = M₂ · V₂"

#### formula_pH_definition
- **canonical_id**: `formula_pH_definition`
- **is_a**: `formula_chemistry`
- **rpn_sketch**: `[RECALL H_concentration][GALAXY_LOOKUP star.symbol.log10][TPACK 1][TNOT][STORE pH]`
- **symlinks**: `star.symbol.equal`
- **surface_forms**:
  - en: "pH = −log₁₀ [H⁺]"
  - pt: "pH = −log₁₀ [H⁺]"
  - es: "pH = −log₁₀ [H⁺]"
  - fr: "pH = −log₁₀ [H⁺]"
  - de: "pH = −log₁₀ [H⁺]"
  - it: "pH = −log₁₀ [H⁺]"
  - ja: "pH = −log₁₀ [H⁺]"
  - zh: "pH = −log₁₀ [H⁺]"
  - ru: "pH = −log₁₀ [H⁺]"

## Cluster 3.H — Unit Conversions

#### formula_celsius_to_fahrenheit
- **canonical_id**: `formula_celsius_to_fahrenheit`
- **is_a**: `formula_unit_conversion`
- **rpn_sketch**: `[RECALL C][GALAXY_LOOKUP star.constant.nine][TMUL][GALAXY_LOOKUP star.constant.five][GALAXY_LOOKUP star.constant.reciprocal][TMUL][GALAXY_LOOKUP star.constant.thirty_two][TADD][STORE F]`
- **symlinks**: `star.letter.C, star.letter.F, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "F = (9/5) · C + 32"
  - pt: "F = (9/5) · C + 32"
  - es: "F = (9/5) · C + 32"
  - fr: "F = (9/5) · C + 32"
  - de: "F = (9/5) · C + 32"
  - it: "F = (9/5) · C + 32"
  - ja: "F = (9/5) · C + 32"
  - zh: "F = (9/5) · C + 32"
  - ru: "F = (9/5) · C + 32"

#### formula_celsius_to_kelvin
- **canonical_id**: `formula_celsius_to_kelvin`
- **is_a**: `formula_unit_conversion`
- **rpn_sketch**: `[RECALL C][GALAXY_LOOKUP star.constant.kelvin_offset][TADD][STORE K]`
- **symlinks**: `star.letter.C, star.letter.K, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "K = C + 273.15"
  - pt: "K = C + 273,15"
  - es: "K = C + 273,15"
  - fr: "K = C + 273,15"
  - de: "K = C + 273,15"
  - it: "K = C + 273,15"
  - ja: "K = C + 273.15"
  - zh: "K = C + 273.15"
  - ru: "K = C + 273,15"

---

**Required canonical alias seeds (Batch 9 Slice C extension):**
- letters: `mu`, `sigma`, `lambda`, `rho`, `g`, `f`, `M`, `R`, `T`, `I`, `G`, `K`
- symbols (already requested by Cluster 2 if overlap): `binom`, `factorial`, `complement`, `union`, `intersect`, `implies`, `log10`, `argmax_count`, `power`
- constants: `three`, `five`, `nine`, `thirty_two`, `kelvin_offset`

All other refs reuse aliases already seeded by Batches 8, 8.1, and 9 Slice B.
