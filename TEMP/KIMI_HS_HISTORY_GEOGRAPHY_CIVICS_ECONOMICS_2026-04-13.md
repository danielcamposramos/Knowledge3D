# Kimi Agent Swarm Result
**Task:** Produce K3D meaning-star plan for HIGH-SCHOOL HISTORY + GEOGRAPHY + CIVICS/GOVERNMENT + ECONOMICS, world curriculum, multi-regional. Each = meaning-star with canonical_id, is_a, RPN sketch using ONLY 
**Mode:** thinking
**Time:** 2026-04-13 01:49:36

────────────────────────────────────────────────────────────────

## Sub-Agent A: Construct K3D meaning-star plan for WORLD HISTORY + GEOGRAPHY covering prehistory through 21st century (Paleolithic to COVID-19), including all ancient civilizations (Mesopotamia, Egypt, Indus, China, India, Greece, Rome, Pre-Columbian Americas), Classical-Medieval period (rise of Christianity/Islam, feudalism, Mongol Empire, African/Asian empires), Early Modern (Renaissance, Exploration, Reformation, Scientific Revolution, Enlightenment, American/French/Haitian Revolutions, Napoleon), 19th century (Industrial Revolution, nationalism/unifications, Latin American independence, abolition, US Civil War, Meiji Restoration, colonialism), 20th century (WWI, Russian Revolution, Great Depression, WWII, Holocaust, Cold War, decolonization, civil rights, PRC, Israeli-Palestinian conflict, apartheid), 21st century (9/11, Arab Spring, COVID-19), with mandatory Brazilian national curriculum thread; plus physical geography (continents, oceans, mountains, rivers, lakes, deserts, climates), political geography (195+ countries, capitals, major cities), demographics (population pyramids, urbanization, megac

The user wants a K3D meaning-star plan for WORLD HISTORY + GEOGRAPHY covering prehistory through the 21st century, with a mandatory Brazilian national curriculum thread, plus physical and political geography.

Key constraints:
- K3D format: canonical_id, is_a, RPN sketch using ONLY specific opcodes (STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK)
- 9-language surface_forms (en/pt/es/fr/de/it/ja/zh/ru)
- Bidirectional symlinks
- Flag saudades candidates heavily (locale-specific names)
- Canonical ID patterns: event_<snake>, person_<snake>, place_<snake>, institution_<snake>, concept_<snake>, era_<snake>, doctrine_<snake>, treaty_<snake>, country_<snake>, city_<snake>
- Cross-cluster symlinks to math date-computation, etc.

Sub-Agent A is responsible for WORLD HISTORY + GEOGRAPHY (the sub_task_a described in the parameters).

I need to construct a comprehensive plan/structure showing how these would be represented in K3D, with specific examples covering:
1. Prehistory (Paleolithic/Mesolithic/Neolithic, etc.)
2. Ancient civilizations (Mesopotamia, Egypt, Indus, China, India, Greece, Rome, Pre-Columbian Americas)
3. Classical-Medieval period
4. Early Modern
5. 19th century
6. 20th century
7. 21st century
8. Brazilian national curriculum thread
9. Physical geography (continents, oceans, mountains, rivers, lakes, deserts, climates)
10. Political geography (countries, capitals, cities)
11. Demographics (population pyramids, urbanization, etc.)

For each entry, I need to provide:
- canonical_id (following the snake_case pattern)
- is_a (type)
- RPN sketch using the restricted opcode set
- surface_forms in 9 languages
- saudades flags where appropriate
- symlinks (bidirectional)

The RPN opcodes allowed are:
- STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK

I need to interpret how to represent historical/geographic concepts using these opcodes:
- TQUANT: for temporal quantities (dates, durations)
- TCOMP: for temporal comparison (before/after, during)
- TADD: for temporal addition (date + duration)
- TMUL: for scaling?
- STORE/RECALL: for storing/retrieving values
- GALAXY_LOOKUP: for referencing other entities
- OP_BRANCH: for conditional logic or branching paths (alternate histories, or categorization)
- TPACK/TUNPACK: for bundling/unbundling time periods or geographic coordinates
- TNOT: for negation

Given the complexity, I should provide a representative sample of entries across the categories rather than attempting to list everything (which would be thousands of entries). The user wants a "plan", so a structured template with examples across the domain is appropriate.

Structure:
1. Era/Period definitions (era_paleolithic, era_ancient, etc.)
2. Civilization/Country entities (country_egypt, country_brazil, etc.)
3. Event entities (event_french_revolution, event_brazilian_independence, etc.)
4. Person entities (person_napoleon, person_pedro_i, etc.)
5. Place entities (place_mount_everest, place_amazon_river, etc.)
6. Concept entities (concept_demographic_transition, concept_imperialism, etc.)
7. Treaty/Institution entities (treaty_tordesillas, institution_united_nations, etc.)

For each, show the RPN sketch that captures the temporal and relational logic.

Important: Brazilian thread must be woven throughout, not just at the end. So link colonial events to Brazil, independence, etc.

Saudades candidates: These are terms that are deeply culturally specific and resist translation. I need to flag these aggressively:
- Names of specific battles (Battle of Aljubarrota - PT specific)
- Feudal terminology (Shogun, Samurai in JP context)
- Brazilian specific: Bandeirantes, Ciclo do Ouro, Lei Áurea, Diretas Já, Estado Novo
- Chinese dynasties in Chinese characters
- Russian terms: Duma, Soviet, Perestroika
- Japanese: Meiji Ishin (not just "Meiji Restoration")
- Arabic terms for Islamic concepts
- Indigenous terms for Pre-Columbian (e.g., Tawantinsuyu for Inca Empire)

Surface forms must be provided for 9 languages:
en (English), pt (Portuguese), es (Spanish), fr (French), de (German), it (Italian), ja (Japanese), zh (Chinese), ru (Russian)

For each entry, I should provide a compact representation.

Let me draft some examples:

**Era: Paleolithic**
- canonical_id: era_paleolithic
- is_a: era
- RPN: [TQUANT -2500000 -10000 BCE] [STORE era_paleolithic] -- rough dates 2.5M ya to 10k BCE
- surface_forms: en:Paleolithic|pt:Paleolítico|es:Paleolítico|fr:Paléolithique|de:Paläolithikum|it:Paleolitico|ja:旧石器時代|zh:旧石器时代|ru:палеолит
- saudades: ja:旧石器時代 (conceptual untranslatability of era divisions), zh:旧石器时代

**Era: Ancient Egypt Old Kingdom**
- canonical_id: era_egypt_old_kingdom
- is_a: era
- RPN: [TQUANT -2686 -2181 BCE] [STORE era_egypt_old_kingdom] [GALAXY_LOOKUP place_egypt]
- surface_forms: en:Old Kingdom of Egypt|pt:Reino Antigo do Egito|es:Reino Antiguo de Egipto|fr:Ancien Empire|de:Altes Reich|it:Antico Regno|ja:古王国時代|zh:古埃及古王国时期|ru:Древнее царство
- saudades: zh:古王国 (specific periodization), ja:古王国時代

**Person: Hammurabi**
- canonical_id: person_hammurabi
- is_a: person
- RPN: [TQUANT -1810 -1750 BCE] [STORE person_hammurabi_lifespan] [GALAXY_LOOKUP country_babylon] [TPACK code_law]
- surface_forms: en:Hammurabi|pt:Hamurábi|es:Hammurabi|fr:Hammurabi|de:Hammurapi|it:Hammurabi|ja:ハムラビ|zh:汉谟拉比|ru:Хаммурапи
- saudades: akkadian:𒄩𒄠𒈬𒊏𒁉 (if we had Akkadian, but stick to 9 languages), ja:ハムラビ法典 (Code of Hammurabi is distinct concept)

**Event: Discovery of Brazil**
- canonical_id: event_discovery_brazil_1500
- is_a: event
- RPN: [TQUANT 1500-04-22] [STORE event_discovery_brazil_1500] [GALAXY_LOOKUP person_pedro_alvares_cabral] [GALAXY_LOOKUP place_porto_seguro] [OP_BRANCH arrival_legend_vs_record]
- surface_forms: en:Discovery of Brazil|pt:Descobrimento do Brasil|es:Descubrimiento de Brasil|fr:Découverte du Brésil|de:Entdeckung Brasiliens|it:Scoperta del Brasile|ja:ブラジルの発見|zh:发现巴西|ru:Открытие Бразилии
- saudades: pt:Descobrimento (specific historiographical term implying "discovery" vs "encounter"), pt:Cabral

**Place: Mount Everest**
- canonical_id: place_mount_everest
- is_a: place_mountain
- RPN: [TPACK 27.9881 86.9250 8848.86] [STORE place_mount_everest] [GALAXY_LOOKUP range_himalaya] [GALAXY_LOOKUP country_nepal] [GALAXY_LOOKUP country_china_tibet]
- surface_forms: en:Mount Everest|pt:Monte Everest|es:Monte Everest|fr:Mont Everest|de:Mount Everest|it:Monte Everest|ja:エベレスト|zh:珠穆朗玛峰|ru:Эверест
- saudades: zh:珠穆朗玛峰 (Qomolangma), ne:Sagarmatha, ja:エベレスト (distinct phonology)

**Concept: Demographic Transition**
- canonical_id: concept_demographic_transition
- is_a: concept
- RPN: [STORE concept_demographic_transition] [OP_BRANCH stage_1 stage_2 stage_3 stage_4 stage_5] [TQUANT high_birth high_death] [TCOMP]
- surface_forms: en:Demographic Transition|pt:Transição Demográfica|es:Transición Demográfica|fr:Transition démographique|de:Demografischer Übergang|it:Transizione demografica|ja:人口転換|zh:人口转变|ru:Демографический переход
- saudades: ja:人口転換 (distinct sociological concept with specific Japanese academic usage)

**Country: Brazil**
- canonical_id: country_brazil
- is_a: country
- RPN: [TPACK -73.9 -34.7 -33.7 -5.2] [STORE country_brazil_bbox] [GALAXY_LOOKUP continent_south_america] [GALAXY_LOOKUP capital_brasilia] [GALAXY_LOOKUP event_independence_brazil_1822]
- surface_forms: en:Brazil|pt:Brasil|es:Brasil|fr:Brésil|de:Brasilien|it:Brasile|ja:ブラジル|zh:巴西|ru:Бразилия
- saudades: pt:Brasil (official name República Federativa do Brasil carries legal specificities)

**Event: Lei Áurea (Golden Law - Abolition in Brazil)**
- canonical_id: event_lei_aurea_1888
- is_a: event
- RPN: [TQUANT 1888-05-13] [STORE event_lei_aurea_1888] [GALAXY_LOOKUP institution_brazilian_empire] [GALAXY_LOOKUP person_princess_isabel] [TNOT [GALAXY_LOOKUP event_us_emancipation_proclamation]] ; distinct from US
- surface_forms: en:Lei Áurea|pt:Lei Áurea|es:Ley Áurea|fr:Loi d'Or|de:Goldenes Gesetz|it:Legge Aurea|ja:ゴールデン法|zh:黄金法|ru:Золотой закон
- saudades: pt:Lei Áurea (untouchable - must keep Portuguese), pt:Isabel (title: Princesa Isabel)

**Institution: Congresso Nacional (Brazil)**
- canonical_id: institution_congresso_nacional_brazil
- is_a: institution_legislature
- RPN: [TPACK bicameral] [STORE institution_congresso_nacional_brazil] [GALAXY_LOOKUP chamber_deputies_brazil] [GALAXY_LOOKUP senate_brazil] [OP_BRANCH ordinary_special_legislative_process]
- surface_forms: en:National Congress of Brazil|pt:Congresso Nacional|es:Congreso Nacional de Brasil|fr:Congrès national du Brésil|de:Nationalkongress von Brasilien|it:Congresso Nazionale del Brasile|ja:ブラジル国民議会|zh:巴西国民议会|ru:Национальный конгресс Бразилии
- saudades: pt:Congresso Nacional (constitutional term), pt:Câmara dos Deputados, pt:Senado Federal

I need to organize this into a coherent plan structure. The user wants a "K3D meaning-star plan", which is a structured document showing how the knowledge graph would be constructed.

I'll organize by:
1. Temporal Framework (Eras)
2. Geographic Framework (Physical)
3. Geographic Framework (Political)
4. Historical Events (chronological)
5. People
6. Concepts/Themes
7. Brazilian Thread Integration
8. Saudades Index

For each section, provide the canonical entries with their K3D properties.

Given the instruction to be "thorough but concise", I should provide representative examples for each category rather than exhaustive lists, but ensure all mandatory elements are represented.

Mandatory elements checklist:
- Prehistory (Paleolithic to Iron Age) ✓
- Ancient civilizations (Mesopotamia, Egypt, Indus, China, India, Greece, Rome, Pre-Columbian) ✓
- Classical-Medieval (Christianity/Islam, feudalism, Mongols, African/Asian empires) ✓
- Early Modern (Renaissance, Exploration, Reformation, Scientific Rev, Enlightenment, Revolutions, Napoleon) ✓
- 19th century (Industrial, nationalism, Latin Am independence, abolition, US Civil War, Meiji, colonialism) ✓
- 20th century (WWI, Russian Rev, Depression, WWII, Holocaust, Cold War, decolonization, civil rights, PRC, Israel-Palestine, apartheid) ✓
- 21st century (9/11, Arab Spring, COVID-19) ✓
- Brazilian national curriculum thread ✓
- Physical geography (continents,

────────────────────────────────────────────────────────────────

## Sub-Agent B: Review: Produce K3D meaning-star plan for HIGH-SCHOOL HISTORY + GEOGRAPHY + CIVICS/GOVERNMENT + ECONOMICS, world curriculum, multi-regional. Each = meaning-star with canonical_id, is_a, RPN sketch using ONLY opcodes (STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK), 9-language surface_forms (en/pt/es/fr/de/it/ja/zh/ru), bidirectional symlinks. Flag saudades candidates heavily — history/politics are intensely locale-specific.</parameter>
<parameter name="sub_task_a">WORLD HISTORY + GEOGRAPHY. HISTORY: Prehistory (Paleolithic/Mesolithic/Neolithic, agricultural revolution, Bronze Age, Iron Age). Ancient civilizations: Mesopotamia (Sumer, Akkad, Babylon, Assyria, Persia; Hammurabi, cuneiform), Egypt (Old/Middle/New Kingdom, pyramids, hieroglyphs, pharaohs Khufu/Ramses/Tutankhamun/Cleopatra), Indus Valley (Harappa, Mohenjo-Daro), Ancient China (Xia, Shang, Zhou, Qin, Han; Confucius, Laozi, Sun Tzu, Great Wall, Silk Road), Ancient India (Vedic, Maurya/Ashoka, Gupta, Buddhism/Hinduism origins), Ancient Greece (Minoan, Mycenaean, polis Athens/Sparta, Persian Wars, Peloponnesian War, Alexander, Hellenistic, philosophy Socrates/Plato/Aristotle, democracy origin), Ancient Rome (Republic, Caesar, Empire, Augustus, Pax Romana, Constantine, fall of Western Empire 476 CE, Byzantine continuation), Pre-Columbian Americas (Olmec, Maya, Aztec, Inca, Mississippian). Classical-Medieval: rise of Christianity/Islam (Muhammad, Rashidun Caliphate, Umayyad, Abbasid, Islamic Golden Age, House of Wisdom), Byzantine Empire, Charlemagne and Holy Roman Empire, feudalism, manorial system, Crusades (1095-1291), Black Death, Hundred Years' War, Mongol Empire (Genghis/Kublai Khan), Tang/Song/Yuan/Ming dynasties, Japanese feudalism (shogunate Kamakura/Muromachi/Tokugawa, samurai, bushido), African empires (Ghana, Mali Mansa Musa, Songhai, Great Zimbabwe, Kush/Axum), Khmer/Angkor, Srivijaya, Vietnamese dynasties. Early Modern: Renaissance (Italian Quattrocento, Leonardo/Michelangelo/Raphael, humanism, printing Gutenberg 1440), Age of Exploration (Portuguese Henry the Navigator, Vasco da Gama, Columbus 1492, Magellan circumnavigation, da Cabral Brazil 1500), Protestant Reformation (Luther 95 theses 1517, Calvin, Henry VIII, Catholic Counter-Reformation Trent, religious wars), Scientific Revolution (Copernicus, Galileo, Kepler, Newton), Age of Absolutism (Louis XIV Sun King, Peter the Great, Frederick), Enlightenment (Locke, Voltaire, Rousseau, Montesquieu, Kant), American Revolution 1775-1783 (Declaration 1776, Constitution, Bill of Rights), French Revolution 1789 (Bastille, Jacobins, Reign of Terror, Napoleon, Napoleonic Wars, Congress of Vienna 1815), Haitian Revolution Toussaint. 19th century: Industrial Revolution (steam engine Watt, textile, railways, Bessemer steel), nationalism, unification of Italy (Garibaldi, Cavour 1861) and Germany (Bismarck 1871), Latin American independence (Bolívar, San Martín, Independência Brasil 1822 Dom Pedro I), abolition of slavery (UK 1833, US 13th Amendment 1865, Brazil Lei Áurea 1888), US Civil War 1861-65 (Lincoln, Emancipation, Gettysburg), Meiji Restoration Japan 1868, Scramble for Africa, Berlin Conference 1884, Opium Wars China, Taiping Rebellion, Romantic/realist/impressionist movements, Marxism (Marx, Engels, Communist Manifesto 1848), colonialism/imperialism peak. 20th century: World War I 1914-18 (MAIN causes, trenches, Eastern Front, Lusitania, Treaty of Versailles, League of Nations), Russian Revolution 1917 (Lenin, Bolsheviks, USSR, Stalin), Great Depression 1929, rise of fascism (Mussolini, Hitler, Franco), World War II 1939-45 (invasion of Poland, Blitzkrieg, Pearl Harbor 1941, Holocaust/Shoah, Stalingrad, D-Day 1944, atomic bombs Hiroshima/Nagasaki 1945, UN founding), Cold War (Truman Doctrine, Marshall Plan, NATO 1949, Warsaw Pact, Korean War 1950-53, Cuban Missile Crisis 1962, Vietnam War, Space Race Sputnik 1957/Apollo 11 1969, Berlin Wall 1961/1989, fall of USSR 1991), decolonization (India 1947 Gandhi/Nehru, African independence wave 1950s-60s, Vietnam, Algeria), civil rights movement (MLK, Rosa Parks, Brown v Board), Mao Zedong and PRC 1949, Cultural Revolution, Deng Xiaoping reforms, Iranian Revolution 1979, Israel-Palestine conflict (1948 founding, 1967 Six-Day, 1973 Yom Kippur, intifadas, Oslo), apartheid South Africa/Mandela. 21st century: 9/11, War on Terror Iraq/Afghanistan, 2008 financial crisis, Arab Spring, COVID-19 pandemic, climate activism, tech revolution. Brazilian history (national curriculum): Pré-colonial, Colonial (capitanias, bandeirantes, ciclo da cana/ouro), Independência 1822, Império Dom Pedro I/II, Abolição 1888, Proclamação República 1889, Era Vargas 1930-45, Estado Novo, JK Brasília, ditadura militar 1964-85, Diretas Já, redemocratização, Plano Real, impeachment Collor/Dilma. GEOGRAPHY: physical (continents 7, oceans 5, major mountain ranges: Andes, Rockies, Appalachians, Alps, Himalayas Everest 8849m, Urals, Atlas, Great Rift Valley, Tian Shan; major rivers: Nile, Amazon, Yangtze, Mississippi, Yenisei, Huang He, Ob, Paraná, Congo, Mekong, Niger, Volga, Danube, Rhine, Ganges, Indus, Tigris, Euphrates; major lakes: Caspian, Superior, Victoria, Huron, Michigan, Tanganyika, Baikal, Malawi; deserts: Sahara, Arabian, Gobi, Kalahari, Atacama, Australian outback, Antarctic polar; climates as in cluster 2), political geography (195+ countries with capitals, major cities, population, GDP ranking intro, flags concept), demographics (population pyramids, demographic transition stages 1-5, urbanization, megacities Tokyo/Delhi/Shanghai/São Paulo/Mexico City/Cairo/Mumbai/Beijing/Dhaka/Osaka), migration (push/pull factors, refugees, diaspora), languages world (Indo-European, Sino-Tibetan, Afroasiatic, Austronesian, Niger-Congo families; top languages: Mandarin, Spanish, English, Hindi, Arabic, Portuguese, Bengali, Russian, Japanese), religions world distribution (Christianity, Islam, Hinduism, Buddhism, Judaism, Sikhism, indigenous), economic geography (developed/developing, Global North/South, BRICS, G7/G20, EU, ASEAN, Mercosul, African Union), urban geography (CBD, suburbs, slums favelas, Von Thünen/Burgess/Hoyt models). For each: canonical_id, RPN, symlinks, surface_forms, saudades (MASSIVE — every country/leader/event has locale-specific names, e.g., Genghis Khan = Чингисхан = 成吉思汗 = ジンギスカン).</sub_task_a>
<parameter name="sub_task_b">CIVICS + GOVERNMENT + POLITICAL SCIENCE + ECONOMICS (HS world curriculum). Civics/Government: forms of government (democracy direct/representative, republic, monarchy constitutional/absolute, dictatorship authoritarian/totalitarian, oligarchy, theocracy, anarchy, federalism vs unitary, parliamentary vs presidential vs semi-presidential), US government (Constitution, Bill of Rights 1-10, Amendments 11-27, three branches: Legislative bicameral Senate+House, Executive President/Cabinet, Judicial Supreme Court, checks and balances, federalism, electoral college, Bill/Act legislative process, judicial review Marbury v Madison, major Supreme Court cases: Brown v Board, Roe v Wade, Miranda, Gideon, Obergefell), UK government (Parliament House of Commons/Lords, PM, constitutional monarchy, unwritten constitution), French Fifth Republic (President, PM, Assembly), German Bundestag/Bundesrat/Kanzler, Japanese Diet/Emperor symbolic, Chinese People's Congress/CCP, Russian Duma/President, Brazilian three powers Executivo/Legislativo/Judiciário (STF, Congresso Nacional Câmara+Senado), political parties (left-right spectrum, liberalism, conservatism, socialism, social democracy, communism, fascism, libertarianism, green politics, populism), elections (plurality/FPTP, proportional representation, ranked-choice, single/double round, primary/caucus), human rights (UDHR 1948 30 articles, ICCPR, ICESCR, Geneva Conventions, UN Charter), civil liberties vs civil rights, international organizations (UN: Security Council P5, General Assembly, ECOSOC, ICJ, Secretary-General, agencies WHO/UNESCO/UNICEF/FAO/ILO/WTO/IMF/World Bank; regional: EU, AU, OAS, ASEAN, Arab League; military: NATO), international law, sovereignty, nation-state, citizenship (jus soli/sanguinis), media (four estates, freedom of press, censorship), NGOs, political ideologies, political philosophy (Hobbes social contract/Leviathan, Locke natural rights, Rousseau general will, Montesquieu separation of powers, Mill liberty, Marx class struggle, Rawls justice, Nozick libertarian). Economics (HS world curriculum): microeconomics basics (supply/demand curves, equilibrium P*Q*, surplus/shortage, elasticity price/income/cross PED/YED/XED, consumer/producer surplus, deadweight loss, taxes/subsidies incidence, price ceilings/floors minimum wage rent control, market structures perfect competition/monopoly/monopolistic competition/oligopoly, externalities positive/negative, public goods non-rival non-excludable, free rider, tragedy of commons, information asymmetry, game theory basic: prisoner's dilemma, Nash equilibrium intro), macroeconomics (GDP nominal/real, GDP per capita, GNP, GNI, components C+I+G+NX, growth rate, business cycle expansion/peak/contraction/trough, inflation CPI/PPI/GDP deflator, deflation, hyperinflation, Phillips curve, stagflation, unemployment types frictional/structural/cyclical/seasonal, Okun's law, aggregate demand/supply, monetary policy central banks Fed/ECB/BoJ/PBoC/BoE/BCB, interest rates, discount rate, reserve requirement, open market operations, quantitative easing, fiscal policy expansionary/contractionary, budget deficit/surplus, national debt, Keynesian vs Classical vs Monetarist vs Austrian schools, multiplier effect, crowding out, Laffer curve), international economics (comparative advantage Ricardo, absolute advantage Smith, trade balance, exports/imports, exchange rates fixed/floating, currency appreciation/depreciation, balance of payments current/capital account, tariffs/quotas/subsidies, free trade agreements NAFTA/USMCA/EU/Mercosur/TPP, globalization, protectionism, WTO, Bretton Woods, gold standard history, petrodollar), development economics (HDI Human Development Index, Gini coefficient inequality, poverty line, MDGs/SDGs, developing vs developed, BRICS, resource curse, Dutch disease, dependency theory vs modernization), personal finance HS (budgeting income/expenses/savings, credit scores, loans APR, mortgages, credit cards, investing basics stocks/bonds/mutual funds/ETFs/index/retirement 401k/IRA/Roth, risk-return, diversification, compound interest cross-link to math Cluster 3, retirement planning, insurance types health/auto/home/life, taxes income/property/sales/capital gains, progressive/regressive/flat), key economists (Smith, Ricardo, Marx, Keynes, Friedman, Hayek, Schumpeter, Sen, Piketty). For each: canonical_id, RPN (many of these are relational/categorical rather than formulaic — use TPACK for institutional structures, OP_BRANCH for election mechanisms, GALAXY_LOOKUP for named entities), symlinks, surface_forms (surface forms for country/institution names in all 9 langs), saudades (e.g., "Bundestag", "Gaokao", "Duma", "ENEM" — many political/economic terms are fundamentally untranslatable).</sub_task_b>
<parameter name="context">K3D Phase 7.A.2 HS world-curriculum history/geography/civics/economics seed. Same strict opcode palette. Canonical ID: event_<snake>, person_<snake>, place_<snake>, institution_<snake>, concept_<snake>, era_<snake>, doctrine_<snake>, treaty_<snake>, country_<snake>, city_<snake>. Cross-cluster symlinks: history events link to geography places AND to political concepts AND to economic concepts AND to dates (math date-computation). 9-language surface_forms. Saudades: this domain has the MOST saudades candidates of any cluster — most historical figures, political institutions, and economic systems have locale-specific names that resist periphrasis. Mark aggressively.</parameter>
<parameter name="web_queries">world history high school curriculum comparison, AP world history topics, IB history syllabus, ENEM história tópicos, Gaokao history China topics, geography high school worldwide, AP human geography topics, AP government civics topics, AP macroeconomics microeconomics topics, IB economics syllabus

The user wants me (Sub-Agent B) to produce a K3D meaning-star plan for CIVICS + GOVERNMENT + POLITICAL SCIENCE + ECONOMICS (High School world curriculum). 

Key requirements:
1. Each entry = meaning-star with:
   - canonical_id (using specified prefixes: event_, person_, place_, institution_, concept_, era_, doctrine_, treaty_, country_, city_)
   - is_a
   - RPN sketch using ONLY allowed opcodes: STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK
   - 9-language surface_forms (en/pt/es/fr/de/it/ja/zh/ru)
   - Bidirectional symlinks
   - Flag saudades candidates heavily (especially political/economic terms that are untranslatable)

2. Sub-task B specifically covers:
   - Civics/Government: forms of government, US government structure, UK, France, Germany, Japan, China, Russia, Brazil governments, political parties, elections, human rights, international organizations, citizenship, media, political philosophy
   - Economics: microeconomics (supply/demand, elasticity, market structures, externalities, game theory), macroeconomics (GDP, inflation, unemployment, fiscal/monetary policy), international economics (comparative advantage, trade, exchange rates), development economics (HDI, Gini, etc.), personal finance, key economists

3. Constraints:
   - RPN must use only the specified opcodes
   - Many concepts are relational/categorical rather than formulaic - use TPACK for institutional structures, OP_BRANCH for election mechanisms, GALAXY_LOOKUP for named entities
   - Cross-cluster symlinks to history events, geography places, dates (math), etc.
   - Surface forms in 9 languages
   - Mark saudades aggressively (e.g., "Bundestag", "Gaokao", "Duma", "ENEM")

Let me structure this:

First, I'll organize by major categories:
1. FORMS OF GOVERNMENT (concept_*)
2. POLITICAL SYSTEMS/INSTITUTIONS (institution_*, concept_*)
3. CIVICS/GOVERNMENT STRUCTURES BY COUNTRY (institution_*, country_* specific)
4. POLITICAL THEORY/PHILOSOPHY (doctrine_*, concept_*, person_*)
5. ELECTIONS AND POLITICAL PROCESSES (concept_*, event_*)
6. HUMAN RIGHTS AND LAW (concept_*, treaty_*)
7. INTERNATIONAL ORGANIZATIONS (institution_*)
8. ECONOMICS - MICRO (concept_*)
9. ECONOMICS - MACRO (concept_*, institution_*)
10. ECONOMICS - INTERNATIONAL (concept_*, treaty_*)
11. ECONOMICS - DEVELOPMENT (concept_*)
12. PERSONAL FINANCE (concept_*)
13. KEY ECONOMISTS (person_*)

For RPN, I need to think about how to represent these:
- STORE/RECALL for variable assignment
- GALAXY_LOOKUP for named entities (e.g., "Bundestag", "Federal Reserve")
- OP_BRANCH for conditional/logical branching (e.g., "if presidential system then...")
- TQUANT for quantities
- TCOMP for comparisons
- TADD/TMUL for aggregation/multiplication
- TNOT for negation
- TPACK/TUNPACK for structural composition

Since many are categorical/relational:
- TPACK can bundle components (e.g., branches of government packed into a structure)
- OP_BRANCH can represent decision trees (e.g., electoral systems)
- GALAXY_LOOKUP is crucial for proper nouns that are saudades

Saudades candidates to flag:
- Institution names: Bundestag, Duma, Diet, Knesset, Majlis, Duma, Congresso Nacional, STF, TSE, ENEM, Gaokao, SAT, Baccalauréat, Abitur, Maturità
- Legal terms: Habeas corpus, Certiorari, Mandamus (though some translate)
- Economic terms: specific tax names, local financial instruments
- Political terms: Filibuster (US-specific), Quorum, Gerrymandering, Pork barrel, Lame duck
- Titles: Chancellor, Prime Minister (different meanings in different systems), President vs Premier vs Chairman

Surface forms need to be provided for all 9 languages where applicable. For some political terms, some languages will use transliterations or loanwords.

Let me start generating specific entries. I'll aim for comprehensive coverage but concise format as requested.

Structure per entry:
```
canonical_id: <id>
is_a: <parent>
RPN: <opcode sequence>
surface_forms: {
  en: "...",
  pt: "...",
  es: "...",
  fr: "...",
  de: "...",
  it: "...",
  ja: "...",
  zh: "...",
  ru: "..."
}
saudades: <bool or list of which languages>
symlinks: [<list of related canonical_ids>]
```

Actually, looking at the context, I should provide a structured plan/outline with specific examples rather than exhaustive enumeration of every single country/leader (which would be thousands of entries). The prompt says "Produce K3D meaning-star plan" - so I should provide the schema and representative examples across all domains, with clear indication of how to scale.

Wait, the instruction says "Be thorough but concise — focus on your assigned sub-task." and "Include specific file paths, line numbers, and code references where relevant."

But this is a planning document, not code. So I'll provide a structured plan with specific examples covering all the topics mentioned in sub_task_b, using the required format.

Let me organize:

**Category 1: Forms of Government** (concept_government_...)
- Democracy (direct/representative)
- Republic
- Monarchy (constitutional/absolute)
- Dictatorship (authoritarian/totalitarian)
- Federalism vs Unitary
- Parliamentary vs Presidential vs Semi-presidential

**Category 2: National Government Structures** (institution_gov_...)
- US: Congress (Senate+House), President, Supreme Court
- UK: Parliament (Commons+Lords), PM, Monarch
- France: Fifth Republic structure
- Germany: Bundestag/Bundesrat/Chancellor
- Japan: Diet/Emperor
- China: NPC/CCP structure
- Russia: Duma/President
- Brazil: Executivo/Legislativo/Judiciário (STF, Congresso)

**Category 3: Political Processes** (concept_election_..., concept_legislative_...)
- Electoral systems (FPTP, Proportional, Ranked-choice)
- Legislative process (Bill->Act)
- Judicial review

**Category 4: Rights and Law** (concept_rights_..., treaty_...)
- UDHR, ICCPR, ICESCR
- Civil liberties vs rights

**Category 5: International Organizations** (institution_un_..., institution_eu_...)
- UN structure (Security Council P5, etc.)
- NATO, EU, ASEAN, AU, OAS

**Category 6: Political Philosophy** (doctrine_..., person_philosopher_...)
- Social contract theorists
- Ideologies: Liberalism, Conservatism, Socialism, Communism, Fascism, Libertarianism

**Category 7: Microeconomics** (concept_micro_...)
- Supply/demand
- Elasticity (PED, YED, XED)
- Market structures
- Externalities
- Game theory (Prisoner's dilemma)

**Category 8: Macroeconomics** (concept_macro_...)
- GDP (nominal/real)
- Inflation/Deflation
- Unemployment types
- Fiscal policy
- Monetary policy (central banks)
- Economic schools (Keynesian, Classical, etc.)

**Category 9: International Economics** (concept_trade_...)
- Comparative advantage
- Trade balance
- Exchange rates
- Bretton Woods, WTO

**Category 10: Development Economics** (concept_dev_...)
- HDI, Gini
- BRICS
- Resource curse

**Category 11: Personal Finance** (concept_finance_personal_...)
- Budgeting
- Credit/Loans
- Investing (401k, IRA)
- Insurance
- Taxes

**Category 12: Economists** (person_economist_...)
- Smith, Ricardo, Marx, Keynes, Friedman, etc.

For RPN examples:
- Concept "Supply and Demand Equilibrium": 
  - Could use RECALL supply RECALL demand TCOMP (compare) OP_BRANCH (find equilibrium)
  - Or simpler: TPACK supply demand TCOMP equilibrium_price
  
- Institution "US Congress":
  - TPACK senate house_of_representatives (packed)
  - STORE congress_structure
  
- Election "First Past The Post":
  - RECALL candidates OP_BRANCH plurality_votes STORE winner

Actually, looking at the strict opcode list: STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK

For categorical data like "Forms of Government", I should use:
- GALAXY_LOOKUP for the specific form name (if it's a proper noun with saudades)
- OP_BRANCH to distinguish between types
- TPACK to bundle characteristics

For example, concept_government_democracy:
RPN: GALAXY_LOOKUP democracy_core OP_BRANCH direct_representative TPACK people_sovereignty rule_of_law

But I need to be careful - the RPN should be a valid sequence. Let me think of RPN as Reverse Polish Notation - operands first, then operators.

Example: 
- To represent "Democracy is a form of government where power resides with the people":
  - GALAXY_LOOKUP people GALAXY_LOOKUP power STORE sovereignty_concept
  - GALAXY_LOOKUP government FORM sovereignty_concept TPACK democracy_def
  
Actually, looking at typical K3D style, the RPN sketches are conceptual. They don't need to be executable code but should indicate how the concept is constructed from primitives.

Let me produce the output now. I'll create a comprehensive plan with specific examples for each major area in Sub-Task B.

File paths: Since this is a plan for K3D Phase 7.A.2, I should reference files like `cluster_7_history_civics/k3d_meaning_stars_civics_econ.json` or similar.

Wait, the prompt mentions "Include specific file paths, line numbers, and code references where relevant." This suggests I should format this as if it's code documentation or a structured file plan. But since I'm generating the plan itself, I'll specify the file structure that should be created.

Let me structure the response as:
1. Overview/Schema
2. Civics/Government Section with specific meaning-stars
3. Economics Section with specific meaning-stars
4. Cross-cluster symlink specifications
5. Saudades registry

I'll ensure all 9 languages are covered for surface forms.

Language codes:
- en: English
- pt: Portuguese  
- es: Spanish
- fr: French
- de: German
- it: Italian
- ja: Japanese
- zh: Chinese
- ru: Russian

For saudades-heavy terms, I'll mark them explicitly and explain why (e.g., "Bundestag" is used in English, French, etc., untranslated; "Duma" same; specific exam names like "Gaokao", "ENEM", "SAT").

Let me start drafting specific entries.

Entry 1: Democracy (concept)
Entry 2: Federalism (concept)
Entry 3: US Congress (institution) - saudades: "Congress", "Capitol"
Entry 4: UK Parliament (institution) - saudades: "Westminster", "Parliament" (used across languages)
Entry 5: Bundestag (institution) - MAJOR SAUDADES
Entry 6: Duma (institution) - MAJOR SAUDADES
Entry 7: National People's Congress (institution) - saudades: "NPC", "人大"
Entry 8: First Past The Post (concept) - saudades: "First Past The Post" used in many languages
Entry 9: Proportional Representation (concept)
Entry 10: Supreme Court (institution) - note different meanings in different countries
Entry 11: Judicial Review (concept) - Marbury v Madison linkage
Entry 12: United Nations (institution)
Entry 13: Security Council P5 (institution subgroup)
Entry 14: NATO (institution) - saudades: acronym used universally
Entry 15: European Union (institution)
Entry 16: Social Contract (doctrine)
Entry 17: Separation of Powers (doctrine)
Entry 18: Supply and Demand (concept)
Entry 19: Price Elasticity of Demand (concept)
Entry 20: GDP (concept)
Entry 21: Inflation (concept)
Entry 22: Monetary Policy (concept)
Entry 23: Federal Reserve (institution) - saudades: "Fed", "Federal Reserve"
Entry 24: Comparative Advantage (concept)
Entry 25: HDI (concept) - saudades: "HDI", "IDH" (in Portuguese/Spanish)
Entry 26: 401k (concept) - saudades: specific to US, used globally
Entry 27: Keynesian Economics (doctrine)
Entry 28: Adam Smith (person)

I should provide the full surface form tables for these examples to show the pattern.

For RPN, examples:
- concept_supply_demand: RECALL price RECALL quantity TCOMP OP_BRANCH equilibrium_found STORE market_equilibrium
- institution_bundestag: GALAXY_LOOKUP germany_federal_parliament TPACK lower_house upper_house STORE federal_legislature_germany
- concept_elasticity: RECALL percent_change_quantity TMUL RECALL percent_change_price TNOT TCOMP STORE elasticity_coefficient

Actually, TNOT is logical not. For elasticity which is |(ΔQ/Q)/(ΔP/P)|, I might need:
RECALL delta_q RECALL q TCOMP RECALL delta_p RECALL p TCOMP TCOMP TQUANT elasticity_value STORE

But I should keep it conceptual rather than trying to do actual math with the limited opcodes.

Better approach:
- Use TPACK to construct composite entities (e.g., branches of government

────────────────────────────────────────────────────────────────

## Synthesis

**K3D Phase 7.A.2 Unified Specification: HS World Curriculum (History + Geography + Civics/Government + Economics)**

*Synthesis Agent Output — Codex Implementation Guide*

---

## 1. Schema Architecture & Canonical Typing

**Unified Type Hierarchy** (merge of A/B ontologies):
```
ROOT
├── temporal: era_<snake>, event_<snake>, treaty_<snake>
├── spatial: place_<natural>, country_<ISO3166_alpha3>, city_<geonames_id>
├── institutional: institution_<snake> (govt, economic, educational, judicial)
├── conceptual: concept_<snake>, doctrine_<snake>
├── biographical: person_<snake>
└── meta: cluster_7_* (cross-reference containers)
```

**Conflict Resolution**: Sub-Agent A used `place_<snake>`; Sub-Agent B used `institution_<snake>`. **Resolution**: Maintain both, but enforce bidirectional symlinks via `GALAXY_LOOKUP` reciprocity rules (see §6).

---

## 2. Opcode Semantics (Strict Palette)

| Opcode | Domain Mapping | Usage Pattern |
|--------|---------------|---------------|
| **TQUANT** | Temporal quantification | `[start_date] [end_date] TQUANT` → temporal interval |
| **TADD** | Temporal succession | `[date] [duration] TADD` → new_date |
| **TMUL** | Temporal scaling | `[era_length] [compression_factor] TMUL` → pedagogical_epoch |
| **TCOMP** | Temporal comparison | `[event_a] [event_b] TCOMP` → before/during/after |
| **TPACK** | Structural bundling | `[component_1] ... [component_n] TPACK` → composite_entity |
| **TUNPACK** | Structural de-composition | `[composite] TUNPACK` → components |
| **OP_BRANCH** | Conditional/categorical logic | `[condition] OP_BRANCH [true_path] [false_path]` |
| **TNOT** | Logical negation | `[proposition] TNOT` |
| **STORE** | Canonical registration | `[computed_value] [canonical_id] STORE` |
| **RECALL** | Dereferencing | `[canonical_id] RECALL` |
| **GALAXY_LOOKUP** | Cross-entity linkage | `[target_id] GALAXY_LOOKUP` |

---

## 3. Unified Meaning-Star Templates

### 3.1 Temporal-Institutional Bridge (Synthesis Addition)
**Example**: `event_meiji_restoration_1868` (History) ↔ `institution_meiji_govt` (Civics) ↔ `concept_mercantilism` (Economics)

```json
{
  "canonical_id": "event_meiji_restoration_1868",
  "is_a": "event_constitutional_transformation",
  "RPN": [
    "TQUANT", 1868, 1912, 
    "STORE", "event_meiji_restoration_1868",
    "GALAXY_LOOKUP", "era_empire_japan",
    "GALAXY_LOOKUP", "person_mutsuhito_meiji",
    "GALAXY_LOOKUP", "country_japan",
    "GALAXY_LOOKUP", "concept_restoration_ishin", 
    "TPACK", "fief_abolition", "conscript_army", "railway_nationalization",
    "STORE", "meiji_reforms_bundle",
    "OP_BRANCH", "bureaucratic_absolutism", "constitutional_monarchy_1889"
  ],
  "surface_forms": {
    "en": "Meiji Restoration|Meiji Renewal|明治維新 (loan)",
    "pt": "Restauração Meiji|Revolução Meiji",
    "es": "Restauración Meiji",
    "fr": "Restauration de Meiji",
    "de": "Meiji-Restauration",
    "it": "Rinnovamento Meiji",
    "ja": "明治維新",
    "zh": "明治维新",
    "ru": "Реставрация Мэйдзи|Мэйдзи исин"
  },
  "saudades": ["ja:明治維新", "zh:明治维新", "concept:ishin"],
  "symlinks": {
    "to_geography": ["country_japan", "city_tokyo_edo"],
    "to_civics": ["institution_diet_japan", "concept_constitution_meiji_1889"],
    "to_economics": ["concept_zaibatsu_rise", "doctrine_state_shinto_economy"],
    "to_math": ["date_computation_1868"]
  }
}
```

### 3.2 Institutional-Economic Composite
**Example**: `institution_bundestag` (High saudades — Sub-Agent B emphasis)

```json
{
  "canonical_id": "institution_bundestag_germany",
  "is_a": "institution_lower_house_federal_parliament",
  "RPN": [
    "GALAXY_LOOKUP", "country_germany",
    "TQUANT", 1949, null,
    "STORE", "bundestag_temporal_scope",
    "TPACK", "wahlrecht", "mischwahlsystem", "fünf_prozent_hürde",
    "STORE", "bundestag_structural_features",
    "OP_BRANCH", "regierung_beteiligung", "opposition_kontrolle"
  ],
  "surface_forms": {
    "en": "Bundestag|German Federal Diet",
    "pt": "Bundestag|Parlamento Federal Alemão",
    "es": "Bundestag",
    "fr": "Bundestag",
    "de": "Bundestag",
    "it": "Bundestag",
    "ja": "連邦議会下院|ブンデスターク",
    "zh": "德国联邦议院",
    "ru": "Бундестаг"
  },
  "saudades": ["de:Bundestag", "de:Fünf-Prozent-Hürde", "de:Wahlrecht", "ja:ブンデスターク"],
  "symlinks": {
    "to_history": ["event_german_unification_1871", "event_wende_1989"],
    "to_civics": ["institution_bundesrat", "institution_bundeskanzler"],
    "to_economics": ["concept_soziale_marktwirtschaft"]
  }
}
```

### 3.3 Economic-Geographic Link (Synthesis Addition)
**Example**: `concept_demographic_transition` linking to population pyramids

```json
{
  "canonical_id": "concept_demographic_transition",
  "is_a": "concept_population_dynamics",
  "RPN": [
    "OP_BRANCH", "stage_1_high_stationary", "stage_2_early_expanding", 
    "stage_3_late_expanding", "stage_4_low_stationary", "stage_5_declining",
    "TPACK", "birth_rate", "death_rate", "natural_increase",
    "STORE", "demographic_variables",
    "GALAXY_LOOKUP", "place_sub_saharan_africa",
    "GALAXY_LOOKUP", "place_western_europe",
    "TCOMP"
  ],
  "surface_forms": {
    "en": "Demographic Transition Model|DTM",
    "pt": "Transição Demográfica",
    "es": "Transición Demográfica",
    "fr": "Transition démographique",
    "de": "Demografischer Übergang",
    "it": "Transizione demografica",
    "ja": "人口転換",
    "zh": "人口转变",
    "ru": "Демографический переход"
  },
  "saudades": ["ja:人口転換"],
  "symlinks": {
    "to_geography": ["concept_population_pyramid", "concept_megacity"],
    "to_economics": ["concept_labor_supply", "concept_aging_economy"],
    "to_history": ["event_industrial_revolution", "event_medical_revolution"]
  }
}
```

---

## 4. Domain-Specific Implementation

### 4.1 History + Geography Integration (Sub-Agent A Core)
**File**: `cluster_7/k3d_history_geography.json`

**Saudades-Heavy Entries** (resolved and expanded):
- `event_lei_aurea_1888`: **Tier 1 Saudade** — Portuguese term used across all 9 languages in academic contexts; bind to `person_princess_isabel`, `institution_brazilian_empire`, `concept_abolition_slavery_atlantic`
- `place_himalaya_everest`: Store coordinates via `TPACK [lat] [lon] [elev] TPACK`
- `person_genghis_khan`: Surface forms must include Mongolian script (ᠴᠢᠩᠭᠢᠰ ᠬᠠᠭᠠᠨ) in metadata field `saudades_extended`

**RPN Pattern for Historical Events**:
```
[TQUANT start end] [STORE event_id] 
[GALAXY_LOOKUP geography_place] 
[GALAXY_LOOKUP participants...] 
[OP_BRANCH causal_factors] 
[TADD duration outcome_date]
```

### 4.2 Civics + Government Integration (Sub-Agent B Core)
**File**: `cluster_7/k3d_civics_institutions.json`

**Critical Institutional Saudades**:
| Canonical ID | Saudades Tokens | RPN Pattern |
|--------------|----------------|-------------|
| `institution_duma_russia` | ru:Дума, zh:杜马, ja:ドゥーマ | `[GALAXY_LOOKUP russia] [TQUANT 1905 null] [OP_BRANCH state_duma_federal]` |
| `institution_gaokao` | zh:高考, ja:高考, en:Gaokao (loan) | `[STORE educational_selection] [TQUANT annual] [TPACK mathematics language sciences]` |
| `institution_enem` | pt:ENEM (Exame Nacional do Ensino Médio) | `[GALAXY_LOOKUP brazil_ministry_education] [TQUANT annual] [OP_BRANCH university_selection high_school_evaluation]` |
| `concept_filibuster` | en:filibuster (used in FR, DE academic) | `[institution_us_senate RECALL] [OP_BRANCH unlimited_debate cloture_vote]` |

### 4.3 Economics Integration (Sub-Agent B Core)
**File**: `cluster_7/k3d_economics.json`

**Cross-Cluster Symlinks**:
- `concept_gdp_nominal` → `date_computation_math` (for base year adjustments)
- `concept_comparative_advantage` → `event_industrial_revolution`, `place_united_kingdom`
- `concept_phillips_curve` → `concept_inflation`, `concept_unemployment`

**RPN for Economic Curves**:
```
RECALL unemployment_rate RECALL inflation_rate TCOMP 
OP_BRANCH short_run_phillips long_run_neutrality
STORE concept_phillips_curve_relationship
```

---

## 5. Saudades Taxonomy & Registry

**Tier 1 (Untranslatable Proper Nouns)**: Keep original in all surface forms
- Political: *Duma*, *Bundestag*, *Knesset*, *Diet* (Japan), *Storting*, *Riksdag*
- Economic: *Zaibatsu*, *Chaebol*, *Guanxi* (if included), *Sistema Tributário Nacional* (BR)
- Historical: *Bandeirantes*, *Lei Áurea*, *Diretas Já*, *Perestroika*, *Glasnost*, *Meiji Ishin*

**Tier 2 (Culturally Embedded Concepts)**:
- `concept_saudade_linguistic_untranslatability`: Meta-concept storing the phenomenology of untranslatable terms across the cluster
- `concept_democracy_athenian` vs `concept_democracy_liberal`: Distinguish via `OP_BRANCH`

**Tier 3 (Institutional Specificity)**:
- `concept_401k`: US-specific, used as loanword globally
- `concept_baccalaureat`: French secondary diploma, distinct from `high_school_diploma`

---

## 6. Cross-Cluster Symlink Matrix

| Source Entity | Target Cluster | Relation Type | RPN Implementation |
|--------------|----------------|---------------|-------------------|
| `event_*` | Math (Cluster 3) | Date computation | `[event_date] GALAXY_LOOKUP math_date_arithmetic` |
| `country_*` | Geography | Spatial containment | `[country_bbox] TPACK [capital_city] STORE` |
| `institution_central_bank_*` | Economics | Monetary authority | `[institution_fed] GALAXY_LOOKUP concept_monetary_policy` |
| `person_ruler_*` | Civics | Sovereignty | `[person] TPACK [institution_monarchy] OP_BRANCH absolutism constitutionalism` |
| `treaty_*` | Law/History | Legal obligation | `[treaty_versailles] GALAXY_LOOKUP event_wwii_causation` |
| `concept_market_structure_*` | Economics | Competition theory | `[concept_oligopoly] GALAXY_LOOKUP place_chaebol_korea` |

---

## 7. 9-Language Surface Form Standards

**Encoding**: UTF-8 with CJK/Arabic/Cyrillic support
**Format**: `canonical_form|variant_1|variant_2`

**Example** `country_brazil`:
```json
{
  "en": "Brazil|Federative Republic of Brazil",
  "pt": "Brasil|República Federativa do Brasil",
  "es": "Brasil|República Federativa del Brasil",
  "fr": "Brésil|République fédérative du Brésil",
  "de": "Brasilien|Föderative Republik Brasilien",
  "it": "Brasile|Repubblica Federativa del Brasile",
  "ja": "ブラジル|ブラジル連邦共和国",
  "zh": "巴西|巴西联邦共和国",
  "ru": "Бразилия|Федеративная Республика Бразилия"
}
```

**Saudades Marking**: Add `⚠️` prefix in `ja` and `zh` fields when using phonetic transcriptions of culturally specific terms (e.g., `ja:⚠️ドゥーマ` for Duma).

---

## 8. Implementation File Structure
