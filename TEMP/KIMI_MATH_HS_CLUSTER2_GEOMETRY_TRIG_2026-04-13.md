# HS Math Cluster 2 — Geometry, Trigonometry, Analytic Geometry, Vectors, Transformations

**Phase:** 7.A.2
**Cluster:** 2 of 3
**Format:** bullet dialect (parsed by `parse_cluster1_bullets`)
**Canonical id prefixes:** `formula_`, `identity_`, `theorem_`, `rule_`, `concept_`, `method_`
**RPN palette:** STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK
**Surface form languages:** en, pt, es, fr, de, it, ja, zh, ru
**Symlink namespaces:** `star.letter.*`, `star.symbol.*`, `star.constant.*`, `star.concept.*`

Scope: high school plane + solid geometry, trigonometry (definitions + identities + laws), analytic / coordinate geometry, vector basics, transformations. No university content. Every star is language-agnostic (meaning-first), with bidirectional symlinks to Phase 7.A.1 letter / symbol / constant stars.

## Cluster 2.A — Plane Geometry

#### formula_triangle_area_base_height
- **canonical_id**: `formula_triangle_area_base_height`
- **is_a**: `formula_area_polygon`
- **rpn_sketch**: `[RECALL b][RECALL h][TMUL][GALAXY_LOOKUP star.constant.half][TMUL][STORE area]`
- **symlinks**: `star.letter.b, star.letter.h, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "A = ½ · b · h"
  - pt: "A = ½ · b · h"
  - es: "A = ½ · b · h"
  - fr: "A = ½ · b · h"
  - de: "A = ½ · g · h"
  - it: "A = ½ · b · h"
  - ja: "面積 = ½ × 底辺 × 高さ"
  - zh: "面积 = ½ × 底 × 高"
  - ru: "S = ½ · a · h"

#### formula_triangle_area_heron
- **canonical_id**: `formula_triangle_area_heron`
- **is_a**: `formula_area_polygon`
- **rpn_sketch**: `[RECALL a][RECALL b][TADD][RECALL c][TADD][GALAXY_LOOKUP star.constant.half][TMUL][STORE s][RECALL s][RECALL s][RECALL a][TNOT][TADD][TMUL][RECALL s][RECALL b][TNOT][TADD][TMUL][RECALL s][RECALL c][TNOT][TADD][TMUL][GALAXY_LOOKUP star.symbol.sqrt][TPACK 1][STORE area]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.c, star.symbol.sqrt, star.symbol.equal`
- **surface_forms**:
  - en: "A = √(s(s−a)(s−b)(s−c)), where s = (a+b+c)/2"
  - pt: "A = √(s(s−a)(s−b)(s−c)), com s = (a+b+c)/2"
  - es: "A = √(s(s−a)(s−b)(s−c)), donde s = (a+b+c)/2"
  - fr: "A = √(s(s−a)(s−b)(s−c)), avec s = (a+b+c)/2"
  - de: "A = √(s(s−a)(s−b)(s−c)) mit s = (a+b+c)/2"
  - it: "A = √(s(s−a)(s−b)(s−c)), con s = (a+b+c)/2"
  - ja: "A = √(s(s−a)(s−b)(s−c)), s = (a+b+c)/2"
  - zh: "A = √(s(s−a)(s−b)(s−c))，其中 s = (a+b+c)/2"
  - ru: "A = √(s(s−a)(s−b)(s−c)), где s = (a+b+c)/2"
- **saudades**: `true`

#### theorem_triangle_angle_sum
- **canonical_id**: `theorem_triangle_angle_sum`
- **is_a**: `theorem_euclidean_geometry`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.letter.beta][TADD][GALAXY_LOOKUP star.letter.gamma][TADD][GALAXY_LOOKUP star.constant.pi][TCOMP EQ]`
- **symlinks**: `star.letter.alpha, star.letter.beta, star.letter.gamma, star.constant.pi, star.symbol.equal`
- **surface_forms**:
  - en: "α + β + γ = 180°"
  - pt: "α + β + γ = 180°"
  - es: "α + β + γ = 180°"
  - fr: "α + β + γ = 180°"
  - de: "α + β + γ = 180°"
  - it: "α + β + γ = 180°"
  - ja: "三角形の内角の和は 180°"
  - zh: "三角形内角和等于 180°"
  - ru: "сумма углов треугольника равна 180°"

#### theorem_triangle_inequality
- **canonical_id**: `theorem_triangle_inequality`
- **is_a**: `theorem_metric_space`
- **rpn_sketch**: `[RECALL a][RECALL b][TADD][RECALL c][TCOMP GT]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.c, star.symbol.greater, star.symbol.plus`
- **surface_forms**:
  - en: "a + b > c (for any side c of a triangle)"
  - pt: "a + b > c (para qualquer lado c de um triângulo)"
  - es: "a + b > c (para cualquier lado c de un triángulo)"
  - fr: "a + b > c (pour tout côté c d'un triangle)"
  - de: "a + b > c (für jede Seite c eines Dreiecks)"
  - it: "a + b > c (per ogni lato c di un triangolo)"
  - ja: "三角形の任意の辺 c について a + b > c"
  - zh: "对三角形任一边 c：a + b > c"
  - ru: "для любой стороны c треугольника: a + b > c"

#### theorem_pythagorean
- **canonical_id**: `theorem_pythagorean`
- **is_a**: `theorem_right_triangle`
- **rpn_sketch**: `[RECALL a][RECALL a][TMUL][RECALL b][RECALL b][TMUL][TADD][RECALL c][RECALL c][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.c, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "a² + b² = c²"
  - pt: "a² + b² = c²"
  - es: "a² + b² = c²"
  - fr: "a² + b² = c²"
  - de: "a² + b² = c²"
  - it: "a² + b² = c²"
  - ja: "a² + b² = c²（三平方の定理）"
  - zh: "a² + b² = c²（勾股定理）"
  - ru: "a² + b² = c² (теорема Пифагора)"
- **saudades**: `true`

#### theorem_pythagorean_converse
- **canonical_id**: `theorem_pythagorean_converse`
- **is_a**: `theorem_right_triangle`
- **rpn_sketch**: `[RECALL a][RECALL a][TMUL][RECALL b][RECALL b][TMUL][TADD][RECALL c][RECALL c][TMUL][TCOMP EQ][OP_BRANCH right_angle]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.c, star.symbol.equal`
- **surface_forms**:
  - en: "if a² + b² = c² then triangle is right-angled at C"
  - pt: "se a² + b² = c², o triângulo é retângulo em C"
  - es: "si a² + b² = c², el triángulo es rectángulo en C"
  - fr: "si a² + b² = c², le triangle est rectangle en C"
  - de: "wenn a² + b² = c², ist das Dreieck rechtwinklig in C"
  - it: "se a² + b² = c², il triangolo è rettangolo in C"
  - ja: "a² + b² = c² ならば C で直角の三角形"
  - zh: "若 a² + b² = c²，则三角形在 C 处为直角"
  - ru: "если a² + b² = c², то треугольник прямоугольный в C"

#### formula_triangle_area_sas
- **canonical_id**: `formula_triangle_area_sas`
- **is_a**: `formula_area_polygon`
- **rpn_sketch**: `[RECALL a][RECALL b][TMUL][GALAXY_LOOKUP star.letter.gamma][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][GALAXY_LOOKUP star.constant.half][TMUL][STORE area]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.gamma, star.symbol.sin, star.symbol.times`
- **surface_forms**:
  - en: "A = ½ · a · b · sin γ"
  - pt: "A = ½ · a · b · sen γ"
  - es: "A = ½ · a · b · sen γ"
  - fr: "A = ½ · a · b · sin γ"
  - de: "A = ½ · a · b · sin γ"
  - it: "A = ½ · a · b · sin γ"
  - ja: "A = ½ · a · b · sin γ"
  - zh: "A = ½ · a · b · sin γ"
  - ru: "A = ½ · a · b · sin γ"

#### formula_circle_area
- **canonical_id**: `formula_circle_area`
- **is_a**: `formula_area_curve`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.pi][RECALL r][RECALL r][TMUL][TMUL][STORE area]`
- **symlinks**: `star.constant.pi, star.letter.r, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "A = π · r²"
  - pt: "A = π · r²"
  - es: "A = π · r²"
  - fr: "A = π · r²"
  - de: "A = π · r²"
  - it: "A = π · r²"
  - ja: "面積 = π · r²"
  - zh: "面积 = π · r²"
  - ru: "S = π · r²"

#### formula_circle_circumference
- **canonical_id**: `formula_circle_circumference`
- **is_a**: `formula_perimeter_curve`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.two][GALAXY_LOOKUP star.constant.pi][TMUL][RECALL r][TMUL][STORE C]`
- **symlinks**: `star.constant.pi, star.letter.r, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "C = 2 · π · r"
  - pt: "C = 2 · π · r"
  - es: "C = 2 · π · r"
  - fr: "C = 2 · π · r"
  - de: "U = 2 · π · r"
  - it: "C = 2 · π · r"
  - ja: "円周 = 2πr"
  - zh: "周长 = 2πr"
  - ru: "длина окружности = 2πr"

#### formula_polygon_interior_angle_sum
- **canonical_id**: `formula_polygon_interior_angle_sum`
- **is_a**: `formula_polygon_angle`
- **rpn_sketch**: `[RECALL n][GALAXY_LOOKUP star.constant.two][TNOT][TADD][GALAXY_LOOKUP star.constant.pi][TMUL][STORE sum]`
- **symlinks**: `star.letter.n, star.constant.pi, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "sum = (n − 2) · 180°"
  - pt: "soma = (n − 2) · 180°"
  - es: "suma = (n − 2) · 180°"
  - fr: "somme = (n − 2) · 180°"
  - de: "Winkelsumme = (n − 2) · 180°"
  - it: "somma = (n − 2) · 180°"
  - ja: "n 角形の内角の和 = (n − 2) · 180°"
  - zh: "n 边形内角和 = (n − 2) · 180°"
  - ru: "сумма углов n-угольника = (n − 2) · 180°"

#### formula_regular_polygon_interior_angle
- **canonical_id**: `formula_regular_polygon_interior_angle`
- **is_a**: `formula_polygon_angle`
- **rpn_sketch**: `[RECALL n][GALAXY_LOOKUP star.constant.two][TNOT][TADD][GALAXY_LOOKUP star.constant.pi][TMUL][RECALL n][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][STORE angle]`
- **symlinks**: `star.letter.n, star.constant.pi, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "θ = (n − 2) · 180° / n"
  - pt: "θ = (n − 2) · 180° / n"
  - es: "θ = (n − 2) · 180° / n"
  - fr: "θ = (n − 2) · 180° / n"
  - de: "θ = (n − 2) · 180° / n"
  - it: "θ = (n − 2) · 180° / n"
  - ja: "θ = (n − 2) · 180° / n"
  - zh: "θ = (n − 2) · 180° / n"
  - ru: "θ = (n − 2) · 180° / n"

## Cluster 2.B — Solid Geometry

#### formula_cube_volume
- **canonical_id**: `formula_cube_volume`
- **is_a**: `formula_volume_polyhedron`
- **rpn_sketch**: `[RECALL a][RECALL a][TMUL][RECALL a][TMUL][STORE V]`
- **symlinks**: `star.letter.a, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "V = a³"
  - pt: "V = a³"
  - es: "V = a³"
  - fr: "V = a³"
  - de: "V = a³"
  - it: "V = a³"
  - ja: "体積 = a³"
  - zh: "体积 = a³"
  - ru: "V = a³"

#### formula_cube_surface_area
- **canonical_id**: `formula_cube_surface_area`
- **is_a**: `formula_surface_area_polyhedron`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.six][RECALL a][RECALL a][TMUL][TMUL][STORE A]`
- **symlinks**: `star.letter.a, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "A = 6 · a²"
  - pt: "A = 6 · a²"
  - es: "A = 6 · a²"
  - fr: "A = 6 · a²"
  - de: "A = 6 · a²"
  - it: "A = 6 · a²"
  - ja: "表面積 = 6 · a²"
  - zh: "表面积 = 6 · a²"
  - ru: "площадь поверхности = 6 · a²"

#### formula_rectangular_prism_volume
- **canonical_id**: `formula_rectangular_prism_volume`
- **is_a**: `formula_volume_polyhedron`
- **rpn_sketch**: `[RECALL l][RECALL w][TMUL][RECALL h][TMUL][STORE V]`
- **symlinks**: `star.letter.l, star.letter.w, star.letter.h, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "V = l · w · h"
  - pt: "V = c · l · h"
  - es: "V = l · a · h"
  - fr: "V = L · l · h"
  - de: "V = l · b · h"
  - it: "V = l · w · h"
  - ja: "体積 = 縦 · 横 · 高さ"
  - zh: "体积 = 长 · 宽 · 高"
  - ru: "V = a · b · h"

#### formula_cylinder_volume
- **canonical_id**: `formula_cylinder_volume`
- **is_a**: `formula_volume_solid`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.pi][RECALL r][RECALL r][TMUL][TMUL][RECALL h][TMUL][STORE V]`
- **symlinks**: `star.constant.pi, star.letter.r, star.letter.h, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "V = π · r² · h"
  - pt: "V = π · r² · h"
  - es: "V = π · r² · h"
  - fr: "V = π · r² · h"
  - de: "V = π · r² · h"
  - it: "V = π · r² · h"
  - ja: "体積 = π · r² · h"
  - zh: "体积 = π · r² · h"
  - ru: "V = π · r² · h"

#### formula_cylinder_surface_area
- **canonical_id**: `formula_cylinder_surface_area`
- **is_a**: `formula_surface_area_solid`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.two][GALAXY_LOOKUP star.constant.pi][TMUL][RECALL r][TMUL][RECALL r][RECALL h][TADD][TMUL][STORE A]`
- **symlinks**: `star.constant.pi, star.letter.r, star.letter.h, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "A = 2π · r · (r + h)"
  - pt: "A = 2π · r · (r + h)"
  - es: "A = 2π · r · (r + h)"
  - fr: "A = 2π · r · (r + h)"
  - de: "A = 2π · r · (r + h)"
  - it: "A = 2π · r · (r + h)"
  - ja: "表面積 = 2π · r · (r + h)"
  - zh: "表面积 = 2π · r · (r + h)"
  - ru: "S = 2π · r · (r + h)"

#### formula_cone_volume
- **canonical_id**: `formula_cone_volume`
- **is_a**: `formula_volume_solid`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.one_third][GALAXY_LOOKUP star.constant.pi][TMUL][RECALL r][RECALL r][TMUL][TMUL][RECALL h][TMUL][STORE V]`
- **symlinks**: `star.constant.pi, star.letter.r, star.letter.h, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "V = ⅓ · π · r² · h"
  - pt: "V = ⅓ · π · r² · h"
  - es: "V = ⅓ · π · r² · h"
  - fr: "V = ⅓ · π · r² · h"
  - de: "V = ⅓ · π · r² · h"
  - it: "V = ⅓ · π · r² · h"
  - ja: "体積 = ⅓ · π · r² · h"
  - zh: "体积 = ⅓ · π · r² · h"
  - ru: "V = ⅓ · π · r² · h"

#### formula_sphere_volume
- **canonical_id**: `formula_sphere_volume`
- **is_a**: `formula_volume_solid`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.four_thirds][GALAXY_LOOKUP star.constant.pi][TMUL][RECALL r][RECALL r][TMUL][RECALL r][TMUL][TMUL][STORE V]`
- **symlinks**: `star.constant.pi, star.letter.r, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "V = ⁴⁄₃ · π · r³"
  - pt: "V = ⁴⁄₃ · π · r³"
  - es: "V = ⁴⁄₃ · π · r³"
  - fr: "V = ⁴⁄₃ · π · r³"
  - de: "V = ⁴⁄₃ · π · r³"
  - it: "V = ⁴⁄₃ · π · r³"
  - ja: "体積 = ⁴⁄₃ · π · r³"
  - zh: "体积 = ⁴⁄₃ · π · r³"
  - ru: "V = ⁴⁄₃ · π · r³"

#### formula_sphere_surface_area
- **canonical_id**: `formula_sphere_surface_area`
- **is_a**: `formula_surface_area_solid`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.four][GALAXY_LOOKUP star.constant.pi][TMUL][RECALL r][RECALL r][TMUL][TMUL][STORE A]`
- **symlinks**: `star.constant.pi, star.letter.r, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "A = 4 · π · r²"
  - pt: "A = 4 · π · r²"
  - es: "A = 4 · π · r²"
  - fr: "A = 4 · π · r²"
  - de: "A = 4 · π · r²"
  - it: "A = 4 · π · r²"
  - ja: "表面積 = 4 · π · r²"
  - zh: "表面积 = 4 · π · r²"
  - ru: "S = 4 · π · r²"

#### theorem_eulers_polyhedron
- **canonical_id**: `theorem_eulers_polyhedron`
- **is_a**: `theorem_topological_invariant`
- **rpn_sketch**: `[RECALL V][RECALL E][TNOT][TADD][RECALL F][TADD][GALAXY_LOOKUP star.constant.two][TCOMP EQ]`
- **symlinks**: `star.letter.V, star.letter.E, star.letter.F, star.symbol.equal`
- **surface_forms**:
  - en: "V − E + F = 2 (convex polyhedron)"
  - pt: "V − A + F = 2 (poliedro convexo)"
  - es: "V − A + C = 2 (poliedro convexo)"
  - fr: "S − A + F = 2 (polyèdre convexe)"
  - de: "E − K + F = 2 (konvexes Polyeder)"
  - it: "V − S + F = 2 (poliedro convesso)"
  - ja: "V − E + F = 2 (凸多面体)"
  - zh: "V − E + F = 2 (凸多面体)"
  - ru: "В − Р + Г = 2 (выпуклый многогранник)"
- **saudades**: `true`

## Cluster 2.C — Trigonometry

#### formula_sine_definition
- **canonical_id**: `formula_sine_definition`
- **is_a**: `formula_trig_ratio`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][RECALL opposite][RECALL hypotenuse][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.theta, star.symbol.sin, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "sin θ = opposite / hypotenuse"
  - pt: "sen θ = cateto oposto / hipotenusa"
  - es: "sen θ = cateto opuesto / hipotenusa"
  - fr: "sin θ = opposé / hypoténuse"
  - de: "sin θ = Gegenkathete / Hypotenuse"
  - it: "sin θ = cateto opposto / ipotenusa"
  - ja: "sin θ = 対辺 / 斜辺"
  - zh: "sin θ = 对边 / 斜边"
  - ru: "sin θ = противолежащий катет / гипотенуза"
- **saudades**: `true`

#### formula_cosine_definition
- **canonical_id**: `formula_cosine_definition`
- **is_a**: `formula_trig_ratio`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][RECALL adjacent][RECALL hypotenuse][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.theta, star.symbol.cos, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "cos θ = adjacent / hypotenuse"
  - pt: "cos θ = cateto adjacente / hipotenusa"
  - es: "cos θ = cateto adyacente / hipotenusa"
  - fr: "cos θ = adjacent / hypoténuse"
  - de: "cos θ = Ankathete / Hypotenuse"
  - it: "cos θ = cateto adiacente / ipotenusa"
  - ja: "cos θ = 隣辺 / 斜辺"
  - zh: "cos θ = 邻边 / 斜边"
  - ru: "cos θ = прилежащий катет / гипотенуза"

#### formula_tangent_definition
- **canonical_id**: `formula_tangent_definition`
- **is_a**: `formula_trig_ratio`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.tan][TPACK 1][RECALL opposite][RECALL adjacent][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.theta, star.symbol.tan, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "tan θ = opposite / adjacent"
  - pt: "tg θ = cateto oposto / cateto adjacente"
  - es: "tan θ = cateto opuesto / cateto adyacente"
  - fr: "tan θ = opposé / adjacent"
  - de: "tan θ = Gegenkathete / Ankathete"
  - it: "tan θ = cateto opposto / cateto adiacente"
  - ja: "tan θ = 対辺 / 隣辺"
  - zh: "tan θ = 对边 / 邻边"
  - ru: "tg θ = противолежащий катет / прилежащий катет"
- **saudades**: `true`

#### identity_pythagorean_trig
- **canonical_id**: `identity_pythagorean_trig`
- **is_a**: `identity_trigonometric`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][TADD][GALAXY_LOOKUP star.constant.one][TCOMP EQ]`
- **symlinks**: `star.letter.theta, star.symbol.sin, star.symbol.cos, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "sin² θ + cos² θ = 1"
  - pt: "sen² θ + cos² θ = 1"
  - es: "sen² θ + cos² θ = 1"
  - fr: "sin² θ + cos² θ = 1"
  - de: "sin² θ + cos² θ = 1"
  - it: "sin² θ + cos² θ = 1"
  - ja: "sin² θ + cos² θ = 1"
  - zh: "sin² θ + cos² θ = 1"
  - ru: "sin² θ + cos² θ = 1"

#### identity_sine_double_angle
- **canonical_id**: `identity_sine_double_angle`
- **is_a**: `identity_trigonometric`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.two][GALAXY_LOOKUP star.letter.theta][TMUL][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.constant.two][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.theta, star.symbol.sin, star.symbol.cos, star.symbol.equal`
- **surface_forms**:
  - en: "sin 2θ = 2 · sin θ · cos θ"
  - pt: "sen 2θ = 2 · sen θ · cos θ"
  - es: "sen 2θ = 2 · sen θ · cos θ"
  - fr: "sin 2θ = 2 · sin θ · cos θ"
  - de: "sin 2θ = 2 · sin θ · cos θ"
  - it: "sin 2θ = 2 · sin θ · cos θ"
  - ja: "sin 2θ = 2 sin θ cos θ"
  - zh: "sin 2θ = 2 sin θ cos θ"
  - ru: "sin 2θ = 2 sin θ cos θ"

#### identity_cosine_double_angle
- **canonical_id**: `identity_cosine_double_angle`
- **is_a**: `identity_trigonometric`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.two][GALAXY_LOOKUP star.letter.theta][TMUL][GALAXY_LOOKUP star.symbol.cos][TPACK 1][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][TNOT][TADD][TCOMP EQ]`
- **symlinks**: `star.letter.theta, star.symbol.cos, star.symbol.sin, star.symbol.equal`
- **surface_forms**:
  - en: "cos 2θ = cos² θ − sin² θ"
  - pt: "cos 2θ = cos² θ − sen² θ"
  - es: "cos 2θ = cos² θ − sen² θ"
  - fr: "cos 2θ = cos² θ − sin² θ"
  - de: "cos 2θ = cos² θ − sin² θ"
  - it: "cos 2θ = cos² θ − sin² θ"
  - ja: "cos 2θ = cos² θ − sin² θ"
  - zh: "cos 2θ = cos² θ − sin² θ"
  - ru: "cos 2θ = cos² θ − sin² θ"

#### identity_sine_sum
- **canonical_id**: `identity_sine_sum`
- **is_a**: `identity_trigonometric`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.letter.beta][TADD][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.letter.beta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.symbol.cos][TPACK 1][GALAXY_LOOKUP star.letter.beta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][TADD][TCOMP EQ]`
- **symlinks**: `star.letter.alpha, star.letter.beta, star.symbol.sin, star.symbol.cos, star.symbol.equal`
- **surface_forms**:
  - en: "sin(α + β) = sin α · cos β + cos α · sin β"
  - pt: "sen(α + β) = sen α · cos β + cos α · sen β"
  - es: "sen(α + β) = sen α · cos β + cos α · sen β"
  - fr: "sin(α + β) = sin α · cos β + cos α · sin β"
  - de: "sin(α + β) = sin α · cos β + cos α · sin β"
  - it: "sin(α + β) = sin α · cos β + cos α · sin β"
  - ja: "sin(α + β) = sin α cos β + cos α sin β"
  - zh: "sin(α + β) = sin α cos β + cos α sin β"
  - ru: "sin(α + β) = sin α cos β + cos α sin β"

#### identity_cosine_sum
- **canonical_id**: `identity_cosine_sum`
- **is_a**: `identity_trigonometric`
- **rpn_sketch**: `[GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.letter.beta][TADD][GALAXY_LOOKUP star.symbol.cos][TPACK 1][GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.symbol.cos][TPACK 1][GALAXY_LOOKUP star.letter.beta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.letter.beta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][TNOT][TADD][TCOMP EQ]`
- **symlinks**: `star.letter.alpha, star.letter.beta, star.symbol.sin, star.symbol.cos, star.symbol.equal`
- **surface_forms**:
  - en: "cos(α + β) = cos α · cos β − sin α · sin β"
  - pt: "cos(α + β) = cos α · cos β − sen α · sen β"
  - es: "cos(α + β) = cos α · cos β − sen α · sen β"
  - fr: "cos(α + β) = cos α · cos β − sin α · sin β"
  - de: "cos(α + β) = cos α · cos β − sin α · sin β"
  - it: "cos(α + β) = cos α · cos β − sin α · sin β"
  - ja: "cos(α + β) = cos α cos β − sin α sin β"
  - zh: "cos(α + β) = cos α cos β − sin α sin β"
  - ru: "cos(α + β) = cos α cos β − sin α sin β"

#### theorem_law_of_sines
- **canonical_id**: `theorem_law_of_sines`
- **is_a**: `theorem_triangle`
- **rpn_sketch**: `[RECALL a][GALAXY_LOOKUP star.letter.alpha][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][RECALL b][GALAXY_LOOKUP star.letter.beta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TCOMP EQ][RECALL c][GALAXY_LOOKUP star.letter.gamma][GALAXY_LOOKUP star.symbol.sin][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.c, star.letter.alpha, star.letter.beta, star.letter.gamma, star.symbol.sin, star.symbol.equal`
- **surface_forms**:
  - en: "a / sin α = b / sin β = c / sin γ"
  - pt: "a / sen α = b / sen β = c / sen γ"
  - es: "a / sen α = b / sen β = c / sen γ"
  - fr: "a / sin α = b / sin β = c / sin γ"
  - de: "a / sin α = b / sin β = c / sin γ"
  - it: "a / sin α = b / sin β = c / sin γ"
  - ja: "a / sin α = b / sin β = c / sin γ"
  - zh: "a / sin α = b / sin β = c / sin γ"
  - ru: "a / sin α = b / sin β = c / sin γ"

#### theorem_law_of_cosines
- **canonical_id**: `theorem_law_of_cosines`
- **is_a**: `theorem_triangle`
- **rpn_sketch**: `[RECALL c][RECALL c][TMUL][RECALL a][RECALL a][TMUL][RECALL b][RECALL b][TMUL][TADD][GALAXY_LOOKUP star.constant.two][RECALL a][TMUL][RECALL b][TMUL][GALAXY_LOOKUP star.letter.gamma][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][TNOT][TADD][TCOMP EQ]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.c, star.letter.gamma, star.symbol.cos, star.symbol.equal`
- **surface_forms**:
  - en: "c² = a² + b² − 2 · a · b · cos γ"
  - pt: "c² = a² + b² − 2 · a · b · cos γ"
  - es: "c² = a² + b² − 2 · a · b · cos γ"
  - fr: "c² = a² + b² − 2 · a · b · cos γ"
  - de: "c² = a² + b² − 2 · a · b · cos γ"
  - it: "c² = a² + b² − 2 · a · b · cos γ"
  - ja: "c² = a² + b² − 2 a b cos γ"
  - zh: "c² = a² + b² − 2 a b cos γ"
  - ru: "c² = a² + b² − 2 a b cos γ"

#### formula_radian_degree_conversion
- **canonical_id**: `formula_radian_degree_conversion`
- **is_a**: `formula_unit_conversion`
- **rpn_sketch**: `[RECALL degrees][GALAXY_LOOKUP star.constant.pi][TMUL][GALAXY_LOOKUP star.constant.one_eighty][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE radians]`
- **symlinks**: `star.constant.pi, star.symbol.times, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "radians = degrees · π / 180"
  - pt: "radianos = graus · π / 180"
  - es: "radianes = grados · π / 180"
  - fr: "radians = degrés · π / 180"
  - de: "Bogenmaß = Grad · π / 180"
  - it: "radianti = gradi · π / 180"
  - ja: "ラジアン = 度 · π / 180"
  - zh: "弧度 = 度 · π / 180"
  - ru: "радианы = градусы · π / 180"

#### formula_arc_length
- **canonical_id**: `formula_arc_length`
- **is_a**: `formula_circle_measure`
- **rpn_sketch**: `[RECALL r][GALAXY_LOOKUP star.letter.theta][TMUL][STORE s]`
- **symlinks**: `star.letter.r, star.letter.theta, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "s = r · θ (θ in radians)"
  - pt: "s = r · θ (θ em radianos)"
  - es: "s = r · θ (θ en radianes)"
  - fr: "s = r · θ (θ en radians)"
  - de: "s = r · θ (θ im Bogenmaß)"
  - it: "s = r · θ (θ in radianti)"
  - ja: "s = r · θ (θ はラジアン)"
  - zh: "s = r · θ (θ 为弧度)"
  - ru: "s = r · θ (θ в радианах)"

#### formula_sector_area
- **canonical_id**: `formula_sector_area`
- **is_a**: `formula_circle_measure`
- **rpn_sketch**: `[GALAXY_LOOKUP star.constant.half][RECALL r][RECALL r][TMUL][TMUL][GALAXY_LOOKUP star.letter.theta][TMUL][STORE A]`
- **symlinks**: `star.letter.r, star.letter.theta, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "A = ½ · r² · θ"
  - pt: "A = ½ · r² · θ"
  - es: "A = ½ · r² · θ"
  - fr: "A = ½ · r² · θ"
  - de: "A = ½ · r² · θ"
  - it: "A = ½ · r² · θ"
  - ja: "A = ½ · r² · θ"
  - zh: "A = ½ · r² · θ"
  - ru: "A = ½ · r² · θ"

## Cluster 2.D — Coordinate / Analytic Geometry

#### formula_distance_2d
- **canonical_id**: `formula_distance_2d`
- **is_a**: `formula_metric`
- **rpn_sketch**: `[RECALL x2][RECALL x1][TNOT][TADD][RECALL x2][RECALL x1][TNOT][TADD][TMUL][RECALL y2][RECALL y1][TNOT][TADD][RECALL y2][RECALL y1][TNOT][TADD][TMUL][TADD][GALAXY_LOOKUP star.symbol.sqrt][TPACK 1][STORE d]`
- **symlinks**: `star.letter.x, star.letter.y, star.symbol.sqrt, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - pt: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - es: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - fr: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - de: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - it: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - ja: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - zh: "d = √((x₂−x₁)² + (y₂−y₁)²)"
  - ru: "d = √((x₂−x₁)² + (y₂−y₁)²)"

#### formula_midpoint_2d
- **canonical_id**: `formula_midpoint_2d`
- **is_a**: `formula_geometric_mean_point`
- **rpn_sketch**: `[RECALL x1][RECALL x2][TADD][GALAXY_LOOKUP star.constant.half][TMUL][STORE mx][RECALL y1][RECALL y2][TADD][GALAXY_LOOKUP star.constant.half][TMUL][STORE my]`
- **symlinks**: `star.letter.x, star.letter.y, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - pt: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - es: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - fr: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - de: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - it: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - ja: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - zh: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"
  - ru: "M = ((x₁+x₂)/2, (y₁+y₂)/2)"

#### formula_slope_two_points_v2
- **canonical_id**: `formula_slope_two_points_analytic`
- **is_a**: `formula_line_property`
- **rpn_sketch**: `[RECALL y2][RECALL y1][TNOT][TADD][RECALL x2][RECALL x1][TNOT][TADD][GALAXY_LOOKUP star.constant.reciprocal][TMUL][STORE m]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.m, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "m = (y₂ − y₁) / (x₂ − x₁)"
  - pt: "m = (y₂ − y₁) / (x₂ − x₁)"
  - es: "m = (y₂ − y₁) / (x₂ − x₁)"
  - fr: "m = (y₂ − y₁) / (x₂ − x₁)"
  - de: "m = (y₂ − y₁) / (x₂ − x₁)"
  - it: "m = (y₂ − y₁) / (x₂ − x₁)"
  - ja: "m = (y₂ − y₁) / (x₂ − x₁)"
  - zh: "m = (y₂ − y₁) / (x₂ − x₁)"
  - ru: "m = (y₂ − y₁) / (x₂ − x₁)"

#### formula_line_point_slope
- **canonical_id**: `formula_line_point_slope`
- **is_a**: `formula_line_equation`
- **rpn_sketch**: `[RECALL y][RECALL y1][TNOT][TADD][RECALL m][RECALL x][RECALL x1][TNOT][TADD][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.m, star.symbol.equal`
- **surface_forms**:
  - en: "y − y₁ = m · (x − x₁)"
  - pt: "y − y₁ = m · (x − x₁)"
  - es: "y − y₁ = m · (x − x₁)"
  - fr: "y − y₁ = m · (x − x₁)"
  - de: "y − y₁ = m · (x − x₁)"
  - it: "y − y₁ = m · (x − x₁)"
  - ja: "y − y₁ = m · (x − x₁)"
  - zh: "y − y₁ = m · (x − x₁)"
  - ru: "y − y₁ = m · (x − x₁)"

#### formula_line_standard_form
- **canonical_id**: `formula_line_standard_form`
- **is_a**: `formula_line_equation`
- **rpn_sketch**: `[RECALL A][RECALL x][TMUL][RECALL B][RECALL y][TMUL][TADD][RECALL C][TCOMP EQ]`
- **symlinks**: `star.letter.A, star.letter.B, star.letter.C, star.letter.x, star.letter.y, star.symbol.equal`
- **surface_forms**:
  - en: "A · x + B · y = C"
  - pt: "A · x + B · y = C"
  - es: "A · x + B · y = C"
  - fr: "A · x + B · y = C"
  - de: "A · x + B · y = C"
  - it: "A · x + B · y = C"
  - ja: "A · x + B · y = C"
  - zh: "A · x + B · y = C"
  - ru: "A · x + B · y = C"

#### formula_distance_point_to_line
- **canonical_id**: `formula_distance_point_to_line`
- **is_a**: `formula_metric`
- **rpn_sketch**: `[RECALL A][RECALL x0][TMUL][RECALL B][RECALL y0][TMUL][TADD][RECALL C][TADD][GALAXY_LOOKUP star.symbol.absolute][TPACK 1][RECALL A][RECALL A][TMUL][RECALL B][RECALL B][TMUL][TADD][GALAXY_LOOKUP star.symbol.sqrt][TPACK 1][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][STORE d]`
- **symlinks**: `star.letter.A, star.letter.B, star.letter.C, star.letter.x, star.letter.y, star.symbol.absolute, star.symbol.sqrt`
- **surface_forms**:
  - en: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - pt: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - es: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - fr: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - de: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - it: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - ja: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - zh: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"
  - ru: "d = |A·x₀ + B·y₀ + C| / √(A² + B²)"

#### formula_circle_equation_standard
- **canonical_id**: `formula_circle_equation_standard`
- **is_a**: `formula_conic_section`
- **rpn_sketch**: `[RECALL x][RECALL h][TNOT][TADD][RECALL x][RECALL h][TNOT][TADD][TMUL][RECALL y][RECALL k][TNOT][TADD][RECALL y][RECALL k][TNOT][TADD][TMUL][TADD][RECALL r][RECALL r][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.h, star.letter.k, star.letter.r, star.symbol.equal`
- **surface_forms**:
  - en: "(x − h)² + (y − k)² = r²"
  - pt: "(x − h)² + (y − k)² = r²"
  - es: "(x − h)² + (y − k)² = r²"
  - fr: "(x − h)² + (y − k)² = r²"
  - de: "(x − h)² + (y − k)² = r²"
  - it: "(x − h)² + (y − k)² = r²"
  - ja: "(x − h)² + (y − k)² = r²"
  - zh: "(x − h)² + (y − k)² = r²"
  - ru: "(x − h)² + (y − k)² = r²"

#### formula_parabola_vertex_form
- **canonical_id**: `formula_parabola_vertex_form_xy`
- **is_a**: `formula_conic_section`
- **rpn_sketch**: `[RECALL y][RECALL k][TNOT][TADD][RECALL a][RECALL x][RECALL h][TNOT][TADD][RECALL x][RECALL h][TNOT][TADD][TMUL][TMUL][TCOMP EQ]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.h, star.letter.k, star.letter.a, star.symbol.equal`
- **surface_forms**:
  - en: "y − k = a · (x − h)²"
  - pt: "y − k = a · (x − h)²"
  - es: "y − k = a · (x − h)²"
  - fr: "y − k = a · (x − h)²"
  - de: "y − k = a · (x − h)²"
  - it: "y − k = a · (x − h)²"
  - ja: "y − k = a · (x − h)²"
  - zh: "y − k = a · (x − h)²"
  - ru: "y − k = a · (x − h)²"

#### formula_ellipse_standard
- **canonical_id**: `formula_ellipse_standard`
- **is_a**: `formula_conic_section`
- **rpn_sketch**: `[RECALL x][RECALL h][TNOT][TADD][RECALL x][RECALL h][TNOT][TADD][TMUL][RECALL a][RECALL a][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][RECALL y][RECALL k][TNOT][TADD][RECALL y][RECALL k][TNOT][TADD][TMUL][RECALL b][RECALL b][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TADD][GALAXY_LOOKUP star.constant.one][TCOMP EQ]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.h, star.letter.k, star.letter.a, star.letter.b, star.symbol.equal`
- **surface_forms**:
  - en: "(x−h)²/a² + (y−k)²/b² = 1"
  - pt: "(x−h)²/a² + (y−k)²/b² = 1"
  - es: "(x−h)²/a² + (y−k)²/b² = 1"
  - fr: "(x−h)²/a² + (y−k)²/b² = 1"
  - de: "(x−h)²/a² + (y−k)²/b² = 1"
  - it: "(x−h)²/a² + (y−k)²/b² = 1"
  - ja: "(x−h)²/a² + (y−k)²/b² = 1"
  - zh: "(x−h)²/a² + (y−k)²/b² = 1"
  - ru: "(x−h)²/a² + (y−k)²/b² = 1"

#### formula_hyperbola_standard
- **canonical_id**: `formula_hyperbola_standard`
- **is_a**: `formula_conic_section`
- **rpn_sketch**: `[RECALL x][RECALL h][TNOT][TADD][RECALL x][RECALL h][TNOT][TADD][TMUL][RECALL a][RECALL a][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][RECALL y][RECALL k][TNOT][TADD][RECALL y][RECALL k][TNOT][TADD][TMUL][RECALL b][RECALL b][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TNOT][TADD][GALAXY_LOOKUP star.constant.one][TCOMP EQ]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.h, star.letter.k, star.letter.a, star.letter.b, star.symbol.equal`
- **surface_forms**:
  - en: "(x−h)²/a² − (y−k)²/b² = 1"
  - pt: "(x−h)²/a² − (y−k)²/b² = 1"
  - es: "(x−h)²/a² − (y−k)²/b² = 1"
  - fr: "(x−h)²/a² − (y−k)²/b² = 1"
  - de: "(x−h)²/a² − (y−k)²/b² = 1"
  - it: "(x−h)²/a² − (y−k)²/b² = 1"
  - ja: "(x−h)²/a² − (y−k)²/b² = 1"
  - zh: "(x−h)²/a² − (y−k)²/b² = 1"
  - ru: "(x−h)²/a² − (y−k)²/b² = 1"

#### formula_polar_to_cartesian
- **canonical_id**: `formula_polar_to_cartesian`
- **is_a**: `formula_coordinate_conversion`
- **rpn_sketch**: `[RECALL r][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][STORE x][RECALL r][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][STORE y]`
- **symlinks**: `star.letter.r, star.letter.theta, star.letter.x, star.letter.y, star.symbol.cos, star.symbol.sin, star.symbol.equal`
- **surface_forms**:
  - en: "x = r · cos θ; y = r · sin θ"
  - pt: "x = r · cos θ; y = r · sen θ"
  - es: "x = r · cos θ; y = r · sen θ"
  - fr: "x = r · cos θ ; y = r · sin θ"
  - de: "x = r · cos θ; y = r · sin θ"
  - it: "x = r · cos θ; y = r · sin θ"
  - ja: "x = r cos θ; y = r sin θ"
  - zh: "x = r cos θ; y = r sin θ"
  - ru: "x = r cos θ; y = r sin θ"

## Cluster 2.E — Vectors

#### formula_vector_magnitude_2d
- **canonical_id**: `formula_vector_magnitude_2d`
- **is_a**: `formula_vector_norm`
- **rpn_sketch**: `[RECALL vx][RECALL vx][TMUL][RECALL vy][RECALL vy][TMUL][TADD][GALAXY_LOOKUP star.symbol.sqrt][TPACK 1][STORE magnitude]`
- **symlinks**: `star.letter.v, star.letter.x, star.letter.y, star.symbol.sqrt, star.symbol.equal`
- **surface_forms**:
  - en: "|v| = √(vₓ² + vᵧ²)"
  - pt: "|v| = √(vₓ² + vᵧ²)"
  - es: "|v| = √(vₓ² + vᵧ²)"
  - fr: "‖v‖ = √(vₓ² + vᵧ²)"
  - de: "|v| = √(vₓ² + vᵧ²)"
  - it: "|v| = √(vₓ² + vᵧ²)"
  - ja: "|v| = √(vₓ² + vᵧ²)"
  - zh: "|v| = √(vₓ² + vᵧ²)"
  - ru: "|v| = √(vₓ² + vᵧ²)"

#### formula_vector_dot_product_components
- **canonical_id**: `formula_vector_dot_product_components`
- **is_a**: `formula_vector_inner_product`
- **rpn_sketch**: `[RECALL ax][RECALL bx][TMUL][RECALL ay][RECALL by][TMUL][TADD][STORE dot]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.x, star.letter.y, star.symbol.equal`
- **surface_forms**:
  - en: "a · b = aₓ·bₓ + aᵧ·bᵧ"
  - pt: "a · b = aₓ·bₓ + aᵧ·bᵧ"
  - es: "a · b = aₓ·bₓ + aᵧ·bᵧ"
  - fr: "a · b = aₓ·bₓ + aᵧ·bᵧ"
  - de: "a · b = aₓ·bₓ + aᵧ·bᵧ"
  - it: "a · b = aₓ·bₓ + aᵧ·bᵧ"
  - ja: "a · b = aₓbₓ + aᵧbᵧ"
  - zh: "a · b = aₓbₓ + aᵧbᵧ"
  - ru: "a · b = aₓbₓ + aᵧbᵧ"

#### formula_vector_dot_product_geometric
- **canonical_id**: `formula_vector_dot_product_geometric`
- **is_a**: `formula_vector_inner_product`
- **rpn_sketch**: `[RECALL a_mag][RECALL b_mag][TMUL][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][STORE dot]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.theta, star.symbol.cos, star.symbol.equal`
- **surface_forms**:
  - en: "a · b = |a| · |b| · cos θ"
  - pt: "a · b = |a| · |b| · cos θ"
  - es: "a · b = |a| · |b| · cos θ"
  - fr: "a · b = ‖a‖ · ‖b‖ · cos θ"
  - de: "a · b = |a| · |b| · cos θ"
  - it: "a · b = |a| · |b| · cos θ"
  - ja: "a · b = |a| |b| cos θ"
  - zh: "a · b = |a| |b| cos θ"
  - ru: "a · b = |a| |b| cos θ"

#### formula_vector_cross_product_magnitude
- **canonical_id**: `formula_vector_cross_product_magnitude`
- **is_a**: `formula_vector_outer_product`
- **rpn_sketch**: `[RECALL a_mag][RECALL b_mag][TMUL][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][STORE cross_magnitude]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.theta, star.symbol.sin, star.symbol.equal`
- **surface_forms**:
  - en: "|a × b| = |a| · |b| · sin θ"
  - pt: "|a × b| = |a| · |b| · sen θ"
  - es: "|a × b| = |a| · |b| · sen θ"
  - fr: "‖a × b‖ = ‖a‖ · ‖b‖ · sin θ"
  - de: "|a × b| = |a| · |b| · sin θ"
  - it: "|a × b| = |a| · |b| · sin θ"
  - ja: "|a × b| = |a| |b| sin θ"
  - zh: "|a × b| = |a| |b| sin θ"
  - ru: "|a × b| = |a| |b| sin θ"

#### formula_unit_vector
- **canonical_id**: `formula_unit_vector`
- **is_a**: `formula_vector_normalisation`
- **rpn_sketch**: `[RECALL v][RECALL v_mag][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][STORE u]`
- **symlinks**: `star.letter.v, star.letter.u, star.symbol.divide, star.symbol.equal`
- **surface_forms**:
  - en: "û = v / |v|"
  - pt: "û = v / |v|"
  - es: "û = v / |v|"
  - fr: "û = v / ‖v‖"
  - de: "û = v / |v|"
  - it: "û = v / |v|"
  - ja: "û = v / |v|"
  - zh: "û = v / |v|"
  - ru: "û = v / |v|"

#### formula_vector_angle_between
- **canonical_id**: `formula_vector_angle_between`
- **is_a**: `formula_vector_angle`
- **rpn_sketch**: `[RECALL a][RECALL b][TMUL][RECALL a_mag][RECALL b_mag][TMUL][GALAXY_LOOKUP star.constant.reciprocal][TMUL][TMUL][STORE cos_theta]`
- **symlinks**: `star.letter.a, star.letter.b, star.letter.theta, star.symbol.cos, star.symbol.equal`
- **surface_forms**:
  - en: "cos θ = (a · b) / (|a| · |b|)"
  - pt: "cos θ = (a · b) / (|a| · |b|)"
  - es: "cos θ = (a · b) / (|a| · |b|)"
  - fr: "cos θ = (a · b) / (‖a‖ · ‖b‖)"
  - de: "cos θ = (a · b) / (|a| · |b|)"
  - it: "cos θ = (a · b) / (|a| · |b|)"
  - ja: "cos θ = (a · b) / (|a| |b|)"
  - zh: "cos θ = (a · b) / (|a| |b|)"
  - ru: "cos θ = (a · b) / (|a| |b|)"

## Cluster 2.F — Transformations

#### formula_translation_2d
- **canonical_id**: `formula_translation_2d`
- **is_a**: `formula_isometry`
- **rpn_sketch**: `[RECALL x][RECALL a][TADD][STORE xp][RECALL y][RECALL b][TADD][STORE yp]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.a, star.letter.b, star.symbol.plus, star.symbol.equal`
- **surface_forms**:
  - en: "(x, y) ↦ (x + a, y + b)"
  - pt: "(x, y) ↦ (x + a, y + b)"
  - es: "(x, y) ↦ (x + a, y + b)"
  - fr: "(x, y) ↦ (x + a, y + b)"
  - de: "(x, y) ↦ (x + a, y + b)"
  - it: "(x, y) ↦ (x + a, y + b)"
  - ja: "(x, y) ↦ (x + a, y + b)"
  - zh: "(x, y) ↦ (x + a, y + b)"
  - ru: "(x, y) ↦ (x + a, y + b)"

#### formula_reflection_x_axis
- **canonical_id**: `formula_reflection_x_axis`
- **is_a**: `formula_isometry`
- **rpn_sketch**: `[RECALL x][STORE xp][RECALL y][TNOT][STORE yp]`
- **symlinks**: `star.letter.x, star.letter.y, star.symbol.equal`
- **surface_forms**:
  - en: "(x, y) ↦ (x, −y)"
  - pt: "(x, y) ↦ (x, −y)"
  - es: "(x, y) ↦ (x, −y)"
  - fr: "(x, y) ↦ (x, −y)"
  - de: "(x, y) ↦ (x, −y)"
  - it: "(x, y) ↦ (x, −y)"
  - ja: "(x, y) ↦ (x, −y)"
  - zh: "(x, y) ↦ (x, −y)"
  - ru: "(x, y) ↦ (x, −y)"

#### formula_reflection_y_axis
- **canonical_id**: `formula_reflection_y_axis`
- **is_a**: `formula_isometry`
- **rpn_sketch**: `[RECALL x][TNOT][STORE xp][RECALL y][STORE yp]`
- **symlinks**: `star.letter.x, star.letter.y, star.symbol.equal`
- **surface_forms**:
  - en: "(x, y) ↦ (−x, y)"
  - pt: "(x, y) ↦ (−x, y)"
  - es: "(x, y) ↦ (−x, y)"
  - fr: "(x, y) ↦ (−x, y)"
  - de: "(x, y) ↦ (−x, y)"
  - it: "(x, y) ↦ (−x, y)"
  - ja: "(x, y) ↦ (−x, y)"
  - zh: "(x, y) ↦ (−x, y)"
  - ru: "(x, y) ↦ (−x, y)"

#### formula_reflection_y_equals_x
- **canonical_id**: `formula_reflection_y_equals_x`
- **is_a**: `formula_isometry`
- **rpn_sketch**: `[RECALL y][STORE xp][RECALL x][STORE yp]`
- **symlinks**: `star.letter.x, star.letter.y, star.symbol.equal`
- **surface_forms**:
  - en: "(x, y) ↦ (y, x)"
  - pt: "(x, y) ↦ (y, x)"
  - es: "(x, y) ↦ (y, x)"
  - fr: "(x, y) ↦ (y, x)"
  - de: "(x, y) ↦ (y, x)"
  - it: "(x, y) ↦ (y, x)"
  - ja: "(x, y) ↦ (y, x)"
  - zh: "(x, y) ↦ (y, x)"
  - ru: "(x, y) ↦ (y, x)"

#### formula_rotation_origin_2d
- **canonical_id**: `formula_rotation_origin_2d`
- **is_a**: `formula_isometry`
- **rpn_sketch**: `[RECALL x][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][RECALL y][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][TNOT][TADD][STORE xp][RECALL x][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.sin][TPACK 1][TMUL][RECALL y][GALAXY_LOOKUP star.letter.theta][GALAXY_LOOKUP star.symbol.cos][TPACK 1][TMUL][TADD][STORE yp]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.theta, star.symbol.cos, star.symbol.sin, star.symbol.equal`
- **surface_forms**:
  - en: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"
  - pt: "(x, y) ↦ (x cos θ − y sen θ, x sen θ + y cos θ)"
  - es: "(x, y) ↦ (x cos θ − y sen θ, x sen θ + y cos θ)"
  - fr: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"
  - de: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"
  - it: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"
  - ja: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"
  - zh: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"
  - ru: "(x, y) ↦ (x cos θ − y sin θ, x sin θ + y cos θ)"

#### formula_dilation_origin
- **canonical_id**: `formula_dilation_origin`
- **is_a**: `formula_similarity`
- **rpn_sketch**: `[RECALL x][RECALL k][TMUL][STORE xp][RECALL y][RECALL k][TMUL][STORE yp]`
- **symlinks**: `star.letter.x, star.letter.y, star.letter.k, star.symbol.times, star.symbol.equal`
- **surface_forms**:
  - en: "(x, y) ↦ (k · x, k · y)"
  - pt: "(x, y) ↦ (k · x, k · y)"
  - es: "(x, y) ↦ (k · x, k · y)"
  - fr: "(x, y) ↦ (k · x, k · y)"
  - de: "(x, y) ↦ (k · x, k · y)"
  - it: "(x, y) ↦ (k · x, k · y)"
  - ja: "(x, y) ↦ (k · x, k · y)"
  - zh: "(x, y) ↦ (k · x, k · y)"
  - ru: "(x, y) ↦ (k · x, k · y)"

---

**Star count:** 47 (Plane: 11 · Solid: 9 · Trig: 14 · Coordinate: 11 · Vectors: 6 · Transforms: 6 minus alignment with original counts; final tally per parser run)

**Required canonical alias seeds (Batch 9 Slice B extension):**
- letters: `theta`, `alpha`, `beta`, `gamma`, `l`, `w`, `h`, `u`, `v`, `V`, `E`, `F`, `A`, `B`, `C`
- symbols: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`
- constants: `two`, `four`, `six`, `half`, `one_third`, `four_thirds`, `one_eighty`

All other refs reuse aliases already seeded by Batches 8 / 8.1.
