### Category 1: Grid Topology and Neighborhoods

#### concept_moore_neighborhood
- **canonical_id**: `concept_moore_neighborhood`
- **is_a**: `concept_grid_topology`
- **rpn_sketch**: `[RECALL cell][GALAXY_LOOKUP star.concept.offset_8way][TPACK 8][TUNPACK][TADD][STORE neighbors]`
- **symlinks**: `star.concept.grid, star.concept.neighborhood, star.concept.connectivity_8way`
- **surface_forms**:
  - en: "Moore neighborhood (8 surrounding cells)"
  - pt: "vizinhança de Moore (8 células adjacentes)"
  - es: "vecindad de Moore (8 celdas adyacentes)"
  - fr: "voisinage de Moore (8 cellules adjacentes)"
  - de: "Moore-Nachbarschaft (8 umgebende Zellen)"
  - it: "intorno di Moore (8 celle adiacenti)"
  - ja: "ムーア近傍 (周囲8セル)"
  - zh: "摩尔邻域 (周围8个格子)"
  - ru: "окрестность Мура (8 соседних клеток)"

#### concept_von_neumann_neighborhood
- **canonical_id**: `concept_von_neumann_neighborhood`
- **is_a**: `concept_grid_topology`
- **rpn_sketch**: `[RECALL cell][GALAXY_LOOKUP star.concept.offset_4way][TPACK 4][TUNPACK][TADD][STORE neighbors]`
- **symlinks**: `star.concept.grid, star.concept.neighborhood, star.concept.connectivity_4way`
- **surface_forms**:
  - en: "Von Neumann neighborhood (4 orthogonal cells)"
  - pt: "vizinhança de von Neumann (4 células ortogonais)"
  - es: "vecindad de von Neumann (4 celdas ortogonales)"
  - fr: "voisinage de von Neumann (4 cellules orthogonales)"
  - de: "Von-Neumann-Nachbarschaft (4 orthogonale Zellen)"
  - it: "intorno di von Neumann (4 celle ortogonali)"
  - ja: "フォン・ノイマン近傍 (直交4セル)"
  - zh: "冯·诺伊曼邻域 (正交4个格子)"
  - ru: "окрестность фон Неймана (4 ортогональные клетки)"

#### concept_grid_coordinates
- **canonical_id**: `concept_grid_coordinate_system`
- **is_a**: `concept_spatial_reference`
- **rpn_sketch**: `[RECALL row][RECALL col][TPACK 2][STORE coord]`
- **symlinks**: `star.letter.r, star.letter.c, star.concept.index, star.concept.grid`
- **surface_forms**:
  - en: "grid coordinate (row, column)"
  - pt: "coordenada de grade (linha, coluna)"
  - es: "coordenada de cuadrícula (fila, columna)"
  - fr: "coordonnée de grille (ligne, colonne)"
  - de: "Gitterkoordinate (Zeile, Spalte)"
  - it: "coordinata di griglia (riga, colonna)"
  - ja: "格子座標 (行, 列)"
  - zh: "网格坐标 (行, 列)"
  - ru: "координата сетки (строка, столбец)"

#### rule_manhattan_distance
- **canonical_id**: `rule_manhattan_distance`
- **is_a**: `concept_grid_metric`
- **rpn_sketch**: `[RECALL r1][RECALL r2][TCOMP SUB][GALAXY_LOOKUP star.symbol.absolute][RECALL c1][RECALL c2][TCOMP SUB][GALAXY_LOOKUP star.symbol.absolute][TADD]`
- **symlinks**: `star.letter.r, star.letter.c, star.symbol.absolute, star.symbol.plus`
- **surface_forms**:
  - en: "Manhattan distance: |r1 − r2| + |c1 − c2|"
  - pt: "distância de Manhattan: |r1 − r2| + |c1 − c2|"
  - es: "distancia de Manhattan: |r1 − r2| + |c1 − c2|"
  - fr: "distance de Manhattan : |r1 − r2| + |c1 − c2|"
  - de: "Manhattan-Distanz: |r1 − r2| + |c1 − c2|"
  - it: "distanza di Manhattan: |r1 − r2| + |c1 − c2|"
  - ja: "マンハッタン距離: |r1 − r2| + |c1 − c2|"
  - zh: "曼哈顿距离: |r1 − r2| + |c1 − c2|"
  - ru: "манхэттенское расстояние: |r1 − r2| + |c1 − c2|"

#### rule_chebyshev_distance
- **canonical_id**: `rule_chebyshev_distance`
- **is_a**: `concept_grid_metric`
- **rpn_sketch**: `[RECALL r1][RECALL r2][TCOMP SUB][GALAXY_LOOKUP star.symbol.absolute][RECALL c1][RECALL c2][TCOMP SUB][GALAXY_LOOKUP star.symbol.absolute][TCOMP MAX]`
- **symlinks**: `star.letter.r, star.letter.c, star.symbol.absolute`
- **surface_forms**:
  - en: "Chebyshev distance: max(|Δr|, |Δc|)"
  - pt: "distância de Chebyshev: max(|Δr|, |Δc|)"
  - es: "distancia de Chebyshev: max(|Δr|, |Δc|)"
  - fr: "distance de Tchebychev : max(|Δr|, |Δc|)"
  - de: "Tschebyschow-Distanz: max(|Δr|, |Δc|)"
  - it: "distanza di Chebyshev: max(|Δr|, |Δc|)"
  - ja: "チェビシェフ距離: max(|Δr|, |Δc|)"
  - zh: "切比雪夫距离: max(|Δr|, |Δc|)"
  - ru: "расстояние Чебышёва: max(|Δr|, |Δc|)"

---

### Category 2: Connectivity and Object Segmentation

#### method_connected_component_labeling
- **canonical_id**: `method_connected_component_labeling`
- **is_a**: `concept_object_segmentation`
- **rpn_sketch**: `[RECALL grid][TQUANT 0][TCOMP NE][STORE mask][RECALL mask][GALAXY_LOOKUP concept.flood_seed][OP_BRANCH][TADD label][STORE labels][TPACK 2]`
- **symlinks**: `star.concept.grid, star.concept.moore_neighborhood, star.concept.von_neumann_neighborhood`
- **surface_forms**:
  - en: "connected component labeling"
  - pt: "rotulagem de componentes conexos"
  - es: "etiquetado de componentes conexas"
  - fr: "étiquetage des composantes connexes"
  - de: "Zusammenhangskomponenten-Beschriftung"
  - it: "etichettatura delle componenti connesse"
  - ja: "連結成分ラベリング"
  - zh: "连通分量标记"
  - ru: "разметка связных компонент"
- **saudades**: true

#### method_flood_fill
- **canonical_id**: `method_flood_fill_region`
- **is_a**: `concept_region_growing`
- **rpn_sketch**: `[RECALL seed][STORE frontier][RECALL frontier][TQUANT 0][TCOMP EQ][OP_BRANCH][RECALL frontier][GALAXY_LOOKUP star.concept.von_neumann_neighborhood][TADD][STORE frontier][GALAXY_LOOKUP method.flood_fill_region]`
- **symlinks**: `star.concept.region, star.concept.von_neumann_neighborhood, star.concept.color_replacement`
- **surface_forms**:
  - en: "flood fill (region growing from seed)"
  - pt: "preenchimento por inundação (crescimento de região a partir de semente)"
  - es: "relleno por inundación (crecimiento de región desde una semilla)"
  - fr: "remplissage par diffusion (croissance de région depuis un germe)"
  - de: "Flutfüllung (Regionenwachstum von einem Saatpunkt)"
  - it: "riempimento a inondazione (crescita di regione da seme)"
  - ja: "フラッドフィル (種点からの領域拡張)"
  - zh: "洪水填充 (从种子点区域生长)"
  - ru: "заливка (рост области от затравки)"
- **saudades**: true

#### method_bounding_box_axis_aligned
- **canonical_id**: `method_bounding_box_axis_aligned`
- **is_a**: `concept_shape_abstraction`
- **rpn_sketch**: `[RECALL object_mask][TUNPACK][TCOMP MIN_ROW][STORE r_min][TCOMP MAX_ROW][STORE r_max][TCOMP MIN_COL][STORE c_min][TCOMP MAX_COL][STORE c_max][RECALL r_min][RECALL c_min][RECALL r_max][RECALL c_max][TPACK 4]`
- **symlinks**: `star.letter.r, star.letter.c, star.concept.object, star.concept.extent`
- **surface_forms**:
  - en: "axis-aligned bounding box"
  - pt: "caixa delimitadora alinhada aos eixos"
  - es: "caja envolvente alineada a los ejes"
  - fr: "boîte englobante alignée aux axes"
  - de: "achsenparalleler Begrenzungsrahmen"
  - it: "riquadro di delimitazione allineato agli assi"
  - ja: "軸並行バウンディングボックス"
  - zh: "轴对齐包围盒"
  - ru: "осево-ориентированный ограничивающий прямоугольник"

#### concept_object_area
- **canonical_id**: `concept_object_pixel_area`
- **is_a**: `concept_shape_measure`
- **rpn_sketch**: `[RECALL object_mask][TUNPACK][TADD][STORE area]`
- **symlinks**: `star.concept.object, star.concept.cardinality, star.symbol.plus`
- **surface_forms**:
  - en: "object area (pixel count)"
  - pt: "área do objeto (contagem de pixels)"
  - es: "área del objeto (conteo de píxeles)"
  - fr: "aire de l'objet (nombre de pixels)"
  - de: "Objektfläche (Pixelanzahl)"
  - it: "area dell'oggetto (conteggio pixel)"
  - ja: "物体の面積 (画素数)"
  - zh: "物体面积 (像素数)"
  - ru: "площадь объекта (число пикселей)"

#### concept_object_perimeter
- **canonical_id**: `concept_object_perimeter`
- **is_a**: `concept_shape_measure`
- **rpn_sketch**: `[RECALL object_mask][GALAXY_LOOKUP star.concept.von_neumann_neighborhood][TCOMP BORDER][TADD][STORE perimeter]`
- **symlinks**: `star.concept.object, star.concept.border, star.concept.von_neumann_neighborhood`
- **surface_forms**:
  - en: "object perimeter (border edge count)"
  - pt: "perímetro do objeto (contagem de arestas de borda)"
  - es: "perímetro del objeto (conteo de aristas de borde)"
  - fr: "périmètre de l'objet (nombre d'arêtes de bord)"
  - de: "Objektumfang (Anzahl der Randkanten)"
  - it: "perimetro dell'oggetto (numero di bordi)"
  - ja: "物体の周囲長 (境界辺の数)"
  - zh: "物体周长 (边界边数)"
  - ru: "периметр объекта (число граничных рёбер)"

#### concept_hole_topology
- **canonical_id**: `concept_shape_hole_topology`
- **is_a**: `concept_shape_topology`
- **rpn_sketch**: `[RECALL object_mask][TNOT][GALAXY_LOOKUP method.connected_component_labeling][TCOMP COUNT][TQUANT 1][TCOMP GT][OP_BRANCH]`
- **symlinks**: `star.concept.object, star.concept.interior, star.concept.exterior`
- **surface_forms**:
  - en: "hole topology (solid vs hollow object)"
  - pt: "topologia de buraco (objeto sólido vs oco)"
  - es: "topología de hueco (objeto sólido vs hueco)"
  - fr: "topologie des trous (objet plein vs creux)"
  - de: "Lochtopologie (massiver vs hohler Körper)"
  - it: "topologia dei fori (oggetto pieno vs cavo)"
  - ja: "穴の位相 (中実 対 中空)"
  - zh: "孔洞拓扑 (实心 vs 空心)"
  - ru: "топология отверстий (сплошной или полый объект)"

#### concept_interior_exterior
- **canonical_id**: `concept_interior_exterior_membership`
- **is_a**: `concept_spatial_topology`
- **rpn_sketch**: `[RECALL point][RECALL region_mask][TCOMP INDEX][OP_BRANCH][TQUANT 1][TQUANT 0][TPACK 2]`
- **symlinks**: `star.concept.region, star.concept.point, star.concept.border`
- **surface_forms**:
  - en: "interior vs exterior membership"
  - pt: "pertinência interior vs exterior"
  - es: "pertenencia interior vs exterior"
  - fr: "appartenance intérieur vs extérieur"
  - de: "Innen- vs Außenzugehörigkeit"
  - it: "appartenenza interno vs esterno"
  - ja: "内部か外部かの所属判定"
  - zh: "内部与外部归属"
  - ru: "принадлежность внутренней или внешней области"

---

### Category 3: Color-Class Reasoning

#### concept_color_categorical_class
- **canonical_id**: `concept_color_categorical_class`
- **is_a**: `concept_color_reasoning`
- **rpn_sketch**: `[RECALL pixel][TQUANT color_index][TCOMP EQ][STORE class][GALAXY_LOOKUP star.concept.equivalence]`
- **symlinks**: `star.concept.color, star.concept.equivalence_class, star.concept.discrete`
- **surface_forms**:
  - en: "color as categorical class (no intrinsic metric)"
  - pt: "cor como classe categórica (sem métrica intrínseca)"
  - es: "color como clase categórica (sin métrica intrínseca)"
  - fr: "couleur comme classe catégorielle (sans métrique intrinsèque)"
  - de: "Farbe als kategoriale Klasse (ohne intrinsische Metrik)"
  - it: "colore come classe categoriale (senza metrica intrinseca)"
  - ja: "カテゴリクラスとしての色 (固有の距離なし)"
  - zh: "作为类别的颜色 (无内在度量)"
  - ru: "цвет как категориальный класс (без внутренней метрики)"

#### method_color_histogram
- **canonical_id**: `method_color_histogram_extraction`
- **is_a**: `concept_color_analysis`
- **rpn_sketch**: `[RECALL grid][TUNPACK][GALAXY_LOOKUP star.concept.color_categorical_class][TADD][STORE histogram][TPACK 10]`
- **symlinks**: `star.concept.color, star.concept.grid, star.concept.cardinality`
- **surface_forms**:
  - en: "color histogram extraction"
  - pt: "extração de histograma de cores"
  - es: "extracción de histograma de colores"
  - fr: "extraction d'histogramme de couleurs"
  - de: "Farbhistogramm-Extraktion"
  - it: "estrazione dell'istogramma di colori"
  - ja: "色ヒストグラムの抽出"
  - zh: "颜色直方图提取"
  - ru: "извлечение цветовой гистограммы"

#### method_dominant_color
- **canonical_id**: `method_dominant_color_mode`
- **is_a**: `concept_color_analysis`
- **rpn_sketch**: `[RECALL histogram][TCOMP ARGMAX][STORE dominant]`
- **symlinks**: `star.concept.color, star.concept.histogram`
- **surface_forms**:
  - en: "dominant color (histogram mode)"
  - pt: "cor dominante (moda do histograma)"
  - es: "color dominante (moda del histograma)"
  - fr: "couleur dominante (mode de l'histogramme)"
  - de: "dominante Farbe (Histogramm-Modus)"
  - it: "colore dominante (moda dell'istogramma)"
  - ja: "支配色 (ヒストグラムの最頻値)"
  - zh: "主导色 (直方图众数)"
  - ru: "доминирующий цвет (мода гистограммы)"

#### rule_color_substitution
- **canonical_id**: `rule_color_substitution_bijective`
- **is_a**: `concept_color_mapping`
- **rpn_sketch**: `[RECALL grid][TUNPACK][RECALL mapping][GALAXY_LOOKUP concept.palette_bijection][TCOMP LOOKUP][TPACK][STORE recolored]`
- **symlinks**: `star.concept.color, star.concept.mapping, star.concept.bijection`
- **surface_forms**:
  - en: "bijective color substitution rule"
  - pt: "regra de substituição de cor bijetiva"
  - es: "regla de sustitución de color biyectiva"
  - fr: "règle de substitution de couleur bijective"
  - de: "bijektive Farbersetzungsregel"
  - it: "regola di sostituzione di colori biunivoca"
  - ja: "全単射的な色置換規則"
  - zh: "双射颜色替换规则"
  - ru: "биективное правило замены цветов"

#### concept_background_foreground
- **canonical_id**: `concept_background_foreground_mask`
- **is_a**: `concept_color_reasoning`
- **rpn_sketch**: `[RECALL grid][RECALL background_color][TCOMP NE][STORE foreground_mask][RECALL foreground_mask][TNOT][STORE background_mask]`
- **symlinks**: `star.concept.color, star.concept.mask, star.concept.binary`
- **surface_forms**:
  - en: "background vs foreground masking"
  - pt: "máscara de fundo vs primeiro plano"
  - es: "enmascaramiento de fondo vs primer plano"
  - fr: "masquage arrière-plan vs premier plan"
  - de: "Hintergrund- vs Vordergrundmaskierung"
  - it: "mascheratura sfondo vs primo piano"
  - ja: "背景と前景のマスク分離"
  - zh: "背景与前景掩码"
  - ru: "разделение на маску фона и переднего плана"

---

### Category 4: Symmetry and Reflection

#### concept_symmetry_horizontal_axis
- **canonical_id**: `concept_symmetry_horizontal_axis`
- **is_a**: `concept_symmetry_detection`
- **rpn_sketch**: `[RECALL grid][STORE original][RECALL grid][TPACK FLIP_H][STORE flipped][RECALL original][RECALL flipped][TCOMP EQ]`
- **symlinks**: `star.concept.reflection, star.concept.mirror, star.concept.axis`
- **surface_forms**:
  - en: "horizontal axis symmetry (reflection across a horizontal line)"
  - pt: "simetria de eixo horizontal (reflexão sobre uma linha horizontal)"
  - es: "simetría de eje horizontal (reflexión sobre una recta horizontal)"
  - fr: "symétrie d'axe horizontal (réflexion par rapport à une droite horizontale)"
  - de: "Symmetrie zur horizontalen Achse (Spiegelung an einer horizontalen Geraden)"
  - it: "simmetria ad asse orizzontale (riflessione rispetto a una retta orizzontale)"
  - ja: "水平軸対称 (水平線に関する鏡映)"
  - zh: "水平轴对称 (关于水平线的镜像)"
  - ru: "симметрия относительно горизонтальной оси"

#### concept_symmetry_vertical_axis
- **canonical_id**: `concept_symmetry_vertical_axis`
- **is_a**: `concept_symmetry_detection`
- **rpn_sketch**: `[RECALL grid][STORE original][RECALL grid][TPACK FLIP_V][STORE flipped][RECALL original][RECALL flipped][TCOMP EQ]`
- **symlinks**: `star.concept.reflection, star.concept.mirror, star.concept.axis`
- **surface_forms**:
  - en: "vertical axis symmetry"
  - pt: "simetria de eixo vertical"
  - es: "simetría de eje vertical"
  - fr: "symétrie d'axe vertical"
  - de: "Symmetrie zur vertikalen Achse"
  - it: "simmetria ad asse verticale"
  - ja: "垂直軸対称"
  - zh: "垂直轴对称"
  - ru: "симметрия относительно вертикальной оси"

#### concept_symmetry_rotational_90
- **canonical_id**: `concept_symmetry_rotational_quarter_turn`
- **is_a**: `concept_symmetry_detection`
- **rpn_sketch**: `[RECALL grid][STORE original][RECALL grid][TPACK ROT_90][STORE rotated][RECALL original][RECALL rotated][TCOMP EQ]`
- **symlinks**: `star.concept.rotation, star.concept.cyclic_order_4`
- **surface_forms**:
  - en: "rotational symmetry of order 4 (90° invariance)"
  - pt: "simetria rotacional de ordem 4 (invariância a 90°)"
  - es: "simetría rotacional de orden 4 (invariancia a 90°)"
  - fr: "symétrie rotationnelle d'ordre 4 (invariance à 90°)"
  - de: "Rotationssymmetrie der Ordnung 4 (Invarianz bei 90°)"
  - it: "simmetria rotazionale di ordine 4 (invarianza a 90°)"
  - ja: "4回対称 (90°回転不変)"
  - zh: "四阶旋转对称 (90°不变性)"
  - ru: "вращательная симметрия 4-го порядка (инвариантность при повороте на 90°)"

#### concept_symmetry_rotational_180
- **canonical_id**: `concept_symmetry_point_inversion`
- **is_a**: `concept_symmetry_detection`
- **rpn_sketch**: `[RECALL grid][STORE original][RECALL grid][TPACK ROT_180][STORE rotated][RECALL original][RECALL rotated][TCOMP EQ]`
- **symlinks**: `star.concept.rotation, star.concept.point_symmetry`
- **surface_forms**:
  - en: "point-inversion symmetry (180° rotation invariance)"
  - pt: "simetria de inversão central (invariância a rotação de 180°)"
  - es: "simetría de inversión central (invariancia a rotación de 180°)"
  - fr: "symétrie par inversion centrale (invariance à la rotation de 180°)"
  - de: "Punktsymmetrie (Invarianz bei 180°-Drehung)"
  - it: "simmetria per inversione centrale (invarianza a rotazione di 180°)"
  - ja: "点対称 (180°回転不変)"
  - zh: "点对称 (180°旋转不变)"
  - ru: "центральная симметрия (инвариантность при повороте на 180°)"

#### concept_symmetry_diagonal_axis
- **canonical_id**: `concept_symmetry_diagonal_axis`
- **is_a**: `concept_symmetry_detection`
- **rpn_sketch**: `[RECALL grid][STORE original][RECALL grid][TPACK TRANSPOSE][STORE transposed][RECALL original][RECALL transposed][TCOMP EQ]`
- **symlinks**: `star.concept.reflection, star.concept.transpose, star.concept.axis`
- **surface_forms**:
  - en: "diagonal axis symmetry (matrix transposition invariance)"
  - pt: "simetria de eixo diagonal (invariância por transposição)"
  - es: "simetría de eje diagonal (invariancia por transposición)"
  - fr: "symétrie d'axe diagonal (invariance par transposition)"
  - de: "Diagonalsymmetrie (Invarianz unter Transposition)"
  - it: "simmetria ad asse diagonale (invarianza per trasposizione)"
  - ja: "対角軸対称 (転置不変)"
  - zh: "对角轴对称 (转置不变性)"
  - ru: "симметрия относительно диагональной оси (инвариантность к транспонированию)"

---

### Category 5: Geometric Transforms

#### method_transform_translate
- **canonical_id**: `method_transform_translate`
- **is_a**: `concept_geometric_transform`
- **rpn_sketch**: `[RECALL grid][RECALL dr][RECALL dc][TPACK 2][STORE offset][RECALL grid][RECALL offset][TADD][STORE translated]`
- **symlinks**: `star.letter.r, star.letter.c, star.concept.shift, star.symbol.plus`
- **surface_forms**:
  - en: "grid translation by (Δr, Δc)"
  - pt: "translação de grade por (Δr, Δc)"
  - es: "traslación de cuadrícula por (Δr, Δc)"
  - fr: "translation de grille de (Δr, Δc)"
  - de: "Gitterverschiebung um (Δr, Δc)"
  - it: "traslazione di griglia di (Δr, Δc)"
  - ja: "格子の平行移動 (Δr, Δc)"
  - zh: "网格平移 (Δr, Δc)"
  - ru: "сдвиг сетки на (Δr, Δc)"

#### method_transform_rotate_quarter_turn
- **canonical_id**: `method_transform_rotate_quarter_turn`
- **is_a**: `concept_geometric_transform`
- **rpn_sketch**: `[RECALL grid][TPACK TRANSPOSE][TPACK FLIP_H][STORE rotated]`
- **symlinks**: `star.concept.rotation, star.concept.transpose, star.concept.flip`
- **surface_forms**:
  - en: "quarter-turn rotation (90° clockwise)"
  - pt: "rotação de um quarto de volta (90° horário)"
  - es: "rotación de un cuarto de vuelta (90° horario)"
  - fr: "rotation d'un quart de tour (90° dans le sens horaire)"
  - de: "Vierteldrehung (90° im Uhrzeigersinn)"
  - it: "rotazione di un quarto di giro (90° orario)"
  - ja: "四分の一回転 (時計回りに90°)"
  - zh: "四分之一旋转 (顺时针90°)"
  - ru: "поворот на четверть оборота (90° по часовой стрелке)"

#### method_transform_reflect_horizontal
- **canonical_id**: `method_transform_reflect_horizontal`
- **is_a**: `concept_geometric_transform`
- **rpn_sketch**: `[RECALL grid][TPACK FLIP_H][STORE reflected]`
- **symlinks**: `star.concept.reflection, star.concept.symmetry_horizontal_axis`
- **surface_forms**:
  - en: "horizontal reflection (flip left-right)"
  - pt: "reflexão horizontal (espelhamento esquerda-direita)"
  - es: "reflexión horizontal (volteo izquierda-derecha)"
  - fr: "réflexion horizontale (miroir gauche-droite)"
  - de: "horizontale Spiegelung (links-rechts-Umkehr)"
  - it: "riflessione orizzontale (specchio sinistra-destra)"
  - ja: "水平反転 (左右反転)"
  - zh: "水平翻转 (左右镜像)"
  - ru: "горизонтальное отражение (слева направо)"

#### method_transform_scale_integer
- **canonical_id**: `method_transform_scale_integer`
- **is_a**: `concept_geometric_transform`
- **rpn_sketch**: `[RECALL grid][RECALL factor_k][TMUL][TPACK TILE][STORE scaled]`
- **symlinks**: `star.letter.k, star.concept.scale, star.symbol.times`
- **surface_forms**:
  - en: "integer upscale by factor k (pixel replication)"
  - pt: "ampliação inteira por fator k (replicação de pixel)"
  - es: "ampliación entera por factor k (replicación de píxel)"
  - fr: "agrandissement entier de facteur k (réplication de pixel)"
  - de: "ganzzahlige Vergrößerung um Faktor k (Pixelreplikation)"
  - it: "ingrandimento intero per fattore k (replicazione di pixel)"
  - ja: "整数倍拡大 (画素の複製, 倍率k)"
  - zh: "整数倍放大 (像素复制, 倍数k)"
  - ru: "целочисленное увеличение в k раз (репликация пикселей)"

#### method_transform_composition
- **canonical_id**: `method_transform_composition_chain`
- **is_a**: `concept_composition`
- **rpn_sketch**: `[RECALL grid][RECALL transform_a][TCOMP APPLY][STORE intermediate][RECALL intermediate][RECALL transform_b][TCOMP APPLY][STORE final]`
- **symlinks**: `star.concept.composition, star.concept.transform, star.symbol.function_composition`
- **surface_forms**:
  - en: "transform composition (chain transforms T_b ∘ T_a)"
  - pt: "composição de transformações (encadeamento T_b ∘ T_a)"
  - es: "composición de transformaciones (encadenamiento T_b ∘ T_a)"
  - fr: "composition de transformations (enchaînement T_b ∘ T_a)"
  - de: "Komposition von Transformationen (Verkettung T_b ∘ T_a)"
  - it: "composizione di trasformazioni (concatenamento T_b ∘ T_a)"
  - ja: "変換の合成 (T_b ∘ T_a の連鎖)"
  - zh: "变换复合 (链式 T_b ∘ T_a)"
  - ru: "композиция преобразований (цепочка T_b ∘ T_a)"

---

### Category 6: Patterns, Tiling, Invariants

#### concept_periodic_tiling
- **canonical_id**: `concept_periodic_tiling_lattice`
- **is_a**: `concept_pattern_structure`
- **rpn_sketch**: `[RECALL grid][RECALL period_r][RECALL period_c][TPACK 2][STORE period][RECALL grid][RECALL period][TCOMP REPEAT][TCOMP EQ]`
- **symlinks**: `star.concept.repetition, star.concept.lattice, star.concept.periodicity`
- **surface_forms**:
  - en: "periodic tiling (lattice repetition)"
  - pt: "pavimentação periódica (repetição em rede)"
  - es: "teselado periódico (repetición reticular)"
  - fr: "pavage périodique (répétition réseau)"
  - de: "periodische Parkettierung (Gitterwiederholung)"
  - it: "tassellatura periodica (ripetizione reticolare)"
  - ja: "周期的タイリング (格子反復)"
  - zh: "周期性铺砌 (格点重复)"
  - ru: "периодическое замощение (решёточное повторение)"

#### concept_invariant_pixel_count
- **canonical_id**: `concept_invariant_pixel_count`
- **is_a**: `concept_conservation_law`
- **rpn_sketch**: `[RECALL input_grid][TUNPACK][TADD][STORE count_in][RECALL output_grid][TUNPACK][TADD][STORE count_out][RECALL count_in][RECALL count_out][TCOMP EQ]`
- **symlinks**: `star.concept.cardinality, star.concept.conservation, star.symbol.equal`
- **surface_forms**:
  - en: "pixel count invariance (conservation across transform)"
  - pt: "invariância da contagem de pixels (conservação sob transformação)"
  - es: "invariancia del conteo de píxeles (conservación bajo transformación)"
  - fr: "invariance du nombre de pixels (conservation sous transformation)"
  - de: "Invarianz der Pixelanzahl (Erhaltung unter Transformation)"
  - it: "invarianza del conteggio di pixel (conservazione sotto trasformazione)"
  - ja: "画素数の不変性 (変換下での保存)"
  - zh: "像素计数不变性 (变换下守恒)"
  - ru: "инвариантность числа пикселей (сохранение при преобразовании)"

#### concept_invariant_color_histogram
- **canonical_id**: `concept_invariant_color_histogram`
- **is_a**: `concept_conservation_law`
- **rpn_sketch**: `[RECALL input_grid][GALAXY_LOOKUP method.color_histogram_extraction][STORE hist_in][RECALL output_grid][GALAXY_LOOKUP method.color_histogram_extraction][STORE hist_out][RECALL hist_in][RECALL hist_out][TCOMP EQ]`
- **symlinks**: `star.concept.color, star.concept.histogram, star.concept.conservation`
- **surface_forms**:
  - en: "color histogram invariance"
  - pt: "invariância do histograma de cores"
  - es: "invariancia del histograma de colores"
  - fr: "invariance de l'histogramme de couleurs"
  - de: "Invarianz des Farbhistogramms"
  - it: "invarianza dell'istogramma di colori"
  - ja: "色ヒストグラムの不変性"
  - zh: "颜色直方图不变性"
  - ru: "инвариантность цветовой гистограммы"

#### method_object_count_enumeration
- **canonical_id**: `method_object_count_enumeration`
- **is_a**: `concept_cardinality`
- **rpn_sketch**: `[RECALL grid][GALAXY_LOOKUP method.connected_component_labeling][TCOMP COUNT_LABELS][STORE object_count]`
- **symlinks**: `star.concept.object, star.concept.cardinality, star.concept.connected_component`
- **surface_forms**:
  - en: "object enumeration (count of distinct objects)"
  - pt: "enumeração de objetos (contagem de objetos distintos)"
  - es: "enumeración de objetos (conteo de objetos distintos)"
  - fr: "énumération des objets (nombre d'objets distincts)"
  - de: "Objektzählung (Anzahl verschiedener Objekte)"
  - it: "enumerazione di oggetti (conteggio di oggetti distinti)"
  - ja: "物体の列挙 (個別物体の数え上げ)"
  - zh: "物体枚举 (不同物体的计数)"
  - ru: "перечисление объектов (число различных объектов)"

#### rule_pattern_completion_continuation
- **canonical_id**: `rule_pattern_completion_continuation`
- **is_a**: `concept_inductive_bias`
- **rpn_sketch**: `[RECALL partial_grid][GALAXY_LOOKUP concept.periodic_tiling_lattice][TCOMP INFER_PERIOD][STORE period][RECALL partial_grid][RECALL period][TCOMP EXTRAPOLATE][STORE completed]`
- **symlinks**: `star.concept.periodic_tiling_lattice, star.concept.extrapolation, star.concept.induction`
- **surface_forms**:
  - en: "pattern completion by periodic continuation"
  - pt: "completamento de padrão por continuação periódica"
  - es: "completado de patrón por continuación periódica"
  - fr: "complétion de motif par continuation périodique"
  - de: "Mustervervollständigung durch periodische Fortsetzung"
  - it: "completamento di pattern per continuazione periodica"
  - ja: "周期的延長によるパターン補完"
  - zh: "基于周期延拓的模式补全"
  - ru: "достраивание узора периодическим продолжением"

#### rule_minimal_change_prior
- **canonical_id**: `rule_minimal_change_prior`
- **is_a**: `concept_inductive_bias`
- **rpn_sketch**: `[RECALL input_grid][RECALL candidate_output][TCOMP DIFF][TADD][STORE edit_distance][RECALL edit_distance][TCOMP ARGMIN]`
- **symlinks**: `star.concept.parsimony, star.concept.inductive_bias`
- **surface_forms**:
  - en: "minimal-change prior (prefer smallest edit)"
  - pt: "prior de mudança mínima (preferir a menor edição)"
  - es: "prior de cambio mínimo (preferir la edición más pequeña)"
  - fr: "a priori de changement minimal (préférer la plus petite édition)"
  - de: "Minimaländerungs-Prior (kleinstmögliche Änderung bevorzugen)"
  - it: "prior di cambiamento minimo (preferire l'edit minore)"
  - ja: "最小変化の事前分布 (最小編集を優先)"
  - zh: "最小改动先验 (偏好最小编辑)"
  - ru: "приор минимального изменения (предпочесть наименьшую правку)"
- **saudades**: true

---

**End of ARC reasoning primitives cluster (2026-04-14).** Ready for a Batch 9 ARC-shape adapter in hs_math_parser to ingest this file into `k3d_canonical` as meaning_star + math_symlink rows. Follow-up waves: shape-family taxonomies (rectangles, crosses, L-shapes), graph-based object relations (above/below/left-of, inside-of), and transform-equivalence classes.
