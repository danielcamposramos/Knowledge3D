# Kimi Agent Swarm Result
**Task:** Produce K3D meaning-star plan for HIGH-SCHOOL HUMANITIES: World Literature + Philosophy/Ethics/Logic + Religious Studies/Comparative Religion + Arts (Visual/Music/Drama/Film). Each = meaning-star with
**Mode:** thinking
**Time:** 2026-04-13 01:52:28

────────────────────────────────────────────────────────────────

## Sub-Agent A: Generate K3D meaning-star plan for WORLD LITERATURE + PHILOSOPHY/ETHICS/LOGIC cluster: literary genres/forms/devices/narrative elements/rhetorical modes/movements (Classical→Medieval→Renaissance→Enlightenment→Romanticism→Realism→Naturalism→Symbolism/Modernism→Latin American Boom→African/Asian/Indian/Arabic literatures), canonical works (Iliad/Odyssey/Aeneid/Divine Comedy/Don Quixote/Shakespeare major plays/Paradise Lost/Crime and Punishment/War and Peace/Ulysses/Gatsby/1984/Mockingbird/Cien Años/Things Fall Apart), philosophy branches/schools (Pre-Socratic→Classical→Hellenistic→Medieval→Eastern→Early Modern→19th century→20th century), ethics systems (consequentialism/deontology/virtue ethics/care ethics/contractarianism/applied), logic formal/informal (propositional/predicate/syllogistic/fallacies/inductive); for each node emit canonical_id, is_a, RPN using ONLY {STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT

**K3D Meaning-Star Plan: WORLD LITERATURE + PHILOSOPHY/ETHICS/LOGIC**  
*Sub-Agent A Output | Phase 7.A.2 HS Humanities Seed | Opcode Palette: {STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK}*

---

### 1. LITERARY GENRES & FORMS

| canonical_id | is_a | RPN Sketch | surface_forms (9-lang) | symlinks | saudades |
|-------------|------|------------|------------------------|----------|----------|
| `genre_epic` | `literary_genre` | `epic_poetry STORE tradition_oral RECALL heroic_narrative TADD TQUANT` | en:epic; pt:épico/epopeia; es:épico/epopeya; fr:épique/épopée; de:Epos/episch; it:epico/epopea; ja:叙事詩 (jojishi); zh:史诗 (shǐshī); ru:эпос/эпопея | `work_iliad`, `work_aeneid`, `work_odyssey` | FALSE |
| `genre_lyric` | `literary_genre` | `subjectivity STORE emotion RECALL musical_origin TADD TQUANT` | en:lyric; pt:lírica; es:lírica; fr:lyrique; de:lyrisch/Lyrik; it:lirica; ja:抒情詩 (jōjōshi); zh:抒情诗 (shūqíngshī); ru:лирика | `poetry_cluster`, `device_sonnet` | FALSE |
| `genre_drama` | `literary_genre` | `theatron STORE performance RECALL dialogue TADD conflict TADD TPACK` | en:drama; pt:drama; es:drama; fr:théâtre/drame; de:Drama; it:dramma; ja:戯曲/演劇 (gekki/engeki); zh:戏剧 (xìjù); ru:драма | `device_dialogue`, `movement_classical_greek` | FALSE |
| `subgenre_tragedy` | `genre_drama` | `hamartia STORE catharsis RECALL fate TADD inevitability TMUL TQUANT` | en:tragedy; pt:tragédia; es:tragedia; fr:tragedie; de:Tragödie; it:tragedia; ja:悲劇 (higeki); zh:悲剧 (bējù); ru:трагедия | `work_oedipus`, `work_hamlet` | FALSE |
| `subgenre_comedy` | `genre_drama` | `humor STORE social_order RECALL happy_resolution TADD TQUANT` | en:comedy; pt:comédia; es:comedia; fr:comédie; de:Komödie; it:commedia; ja:喜劇 (kigeki); zh:喜剧 (xǐjù); ру:комедия | `work_lysistrata`, `author_aristophanes` | FALSE |
| `form_novel` | `literary_form` | `prose STORE extended_narrative RECALL character_development TADD realism_branch OP_BRANCH` | en:novel; pt:romance*; es:novela; fr:roman; de:Roman; it:romanzo; ja:小説 (shōsetsu); zh:小说 (xiǎoshuō); ru:роман | `movement_realism`, `work_war_and_peace` | *PT saudade: "romance" vs "novela"* |
| `form_sonnet_petrarchan` | `poetic_form` | `octave STORE sestet RECALL volta TADD abba_pattern TPACK TQUANT` | en:Petrarchan sonnet; pt:soneto petrarquista; es:soneto petrarquista; fr:sonnet pétrarquiste; de:petrarkistisches Sonett; it:sonetto petrarchesco; ja:ペトラルカ的ソネット; zh:彼特拉克式十四行诗; ru:петраркистский сонет | `author_petrarch`, `device_volta` | FALSE |
| `form_sonnet_shakespearean` | `poetic_form` | `three_quatrains STORE couplet RECALL cdcd_efef_gg TADD TPACK` | en:Shakespearean sonnet; pt:soneto shakesperiano; es:soneto shakesperiano; fr:sonnet shakespearien; de:shakespearesches Sonett; it:sonetto shakespeariano; ja:シェイクスピア的ソネット; zh:莎士比亚式十四行诗; ru:шекспировский сонет | `author_shakespeare`, `device_rhyme_scheme` | FALSE |
| `form_haiku` | `poetic_form` | `5-7-5 STORE kigo RECALL nature_image TADD seasonal_ref TADD TPACK` | en:haiku; pt:haicai/haiku; es:haiku; fr:haïku; de:Haiku; it:haiku; ja:俳句 (haiku); zh:俳句 (páijù); ru:хайку | `concept_kigo`, `author_basho` | **TRUE** (ja structure) |
| `form_ghazal` | `poetic_form` | `couplets STORE radif RECALL qaafiya TADD beloved_divine TADD TPACK` | en:ghazal; pt:gazal; es:gazal; fr:ghazal; de:Ghasel; it:ghazal; ja:ガザル; zh:加扎勒诗; ru:газель | `movement_arabic_literature` | **TRUE** (Perso-Arabic aesthetic) |
| `form_villanelle` | `poetic_form` | `nineteen_lines STORE refrains RECALL A1_b_A2_ab_A1_ab_A2_ab_A1_A2 TPACK` | en:villanelle; pt:villanela; es:villanela; fr:villanelle; de:Villanelle; it:villanella; ja:ヴィラネル; zh:维拉内拉; ru:вилланель | `work_do_not_go_gentle` | FALSE |

---

### 2. LITERARY DEVICES & RHETORIC

| canonical_id | is_a | RPN Sketch | surface_forms | symlinks | saudades |
|-------------|------|------------|---------------|----------|----------|
| `device_metaphor` | `figurative_language` | `tenor STORE vehicle RECALL implicit_comparison TADD TQUANT` | en:metaphor; pt:metáfora; es:metáfora; fr:métaphore; de:Metapher; it:metafora; ja:隠喩 (in'yu); zh:隐喻 (yǐnyù); ru:метафора | `grammar_cluster.simile` | FALSE |
| `device_saudade` | `emotion_concept` | `absence STORE presence RECALL melancholy_longing TADD nostalgia TADD TQUANT` | en:saudade (loan); pt:saudade; es:saudade/añoranza; fr:saudade; de:Saudade/Sehnsucht; it:saudade; ja:サウダージ; zh:萨乌达德; ru:саудаде | `genre_fado`, `concept_melancholy` | **TRUE** (prime) |
| `device_weltschmerz` | `emotion_concept` | `world STORE pain RECALL pessimism TADD romanticism_origin TADD TQUANT` | en:Weltschmerz; pt:Weltschmerz; es:Weltschmerz/pesimismo; fr:mal du siècle; de:Weltschmerz; it:Weltschmerz; ja:ヴェルトシュメルツ; zh:世纪病/厌世; ru:мировая боль | `movement_romanticism` | **TRUE** (de) |
| `device_duende` | `aesthetic_concept` | `soul_STORE darkness_RECALL artistic_daemon TADD flamenco_origin TADD TQUANT` | en:duende; pt:duende; es:duende; fr:duende; de:Duende; it:duende; ja:ドゥエンデ; zh:灵魔/魔力; ru:дуэнди | `movement_symbolism`, `genre_flamenco` | **TRUE** (es) |
| `device_mono_no_aware` | `aesthetic_concept` | `thing_STORE pathos_RECALL impermanence TADD beauty_sadness TADD TQUANT` | en:mono no aware; pt:mono no aware; es:mono no aware; fr:pathétique des choses; de:Mono no aware; it:mono no aware; ja:物の哀れ (もののあはれ); zh:物哀; ru:моно но аварэ | `movement_japanese_classical`, `author_murasaki` | **TRUE** (ja) |
| `device_wabi_sabi` | `aesthetic_concept` | `imperfection_STORE impermanence_RECALL incompleteness TADD rustic_beauty TADD TQUANT` | en:wabi-sabi; pt:wabi-sabi; es:wabi-sabi; fr:wabi-sabi; de:Wabi-Sabi; it:wabi-sabi; ja:侘寂 (わびさび); zh:侘寂; ru:вби-саби | `concept_zen`, `art_cluster` | **TRUE** (ja) |
| `device_metonymy` | `figurative_language` | `container STORE contained_RECALL association TADD substitution TADD TQUANT` | en:metonymy; pt:metonímia; es:metonimia; fr:métonymie; de:Metonymie; it:metonimia; ja:換喩 (kan'yu); zh:转喻 (zhuǎnyù); ru:метонимия | `grammar_cluster.synecdoche` | FALSE |
| `device

────────────────────────────────────────────────────────────────

## Sub-Agent B: Review: Produce K3D meaning-star plan for HIGH-SCHOOL HUMANITIES: World Literature + Philosophy/Ethics/Logic + Religious Studies/Comparative Religion + Arts (Visual/Music/Drama/Film). Each = meaning-star with canonical_id, is_a, RPN sketch using ONLY opcodes (STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK), 9-language surface_forms (en/pt/es/fr/de/it/ja/zh/ru), bidirectional symlinks. Flag saudades candidates heavily.</parameter>
<parameter name="sub_task_a">WORLD LITERATURE + PHILOSOPHY/ETHICS/LOGIC. LITERATURE: literary genres (epic, lyric, drama tragedy/comedy/tragicomedy/satyr, novel, novella, short story, poetry, essay, autobiography, biography, memoir), poetic forms (sonnet Shakespearean/Petrarchan, haiku 5-7-5, tanka, ghazal, villanelle, sestina, ode, elegy, ballad, free verse, blank verse iambic pentameter), literary devices (metaphor, simile, personification, hyperbole, alliteration, assonance, consonance, onomatopoeia, symbolism, allegory, imagery, irony verbal/dramatic/situational, foreshadowing, flashback, allusion, metonymy, synecdoche, oxymoron, paradox, understatement, anaphora, epistrophe, chiasmus, zeugma, caesura, enjambment), narrative elements (plot: exposition/rising action/climax/falling action/resolution Freytag's pyramid, character protagonist/antagonist/foil/round/flat, setting, theme, point of view 1st/2nd/3rd limited/omniscient/unreliable narrator, tone, mood, voice), rhetorical modes (narration, description, exposition, argumentation, persuasion), literary movements/periods: Classical antiquity (Homer Iliad/Odyssey, Sappho, Virgil Aeneid, Ovid, Sophocles Oedipus, Euripides, Aristophanes), Medieval (Beowulf, Dante Divine Comedy, Chaucer Canterbury Tales, 1001 Nights, Tale of Genji Murasaki Shikibu, Journey to the West Wu Cheng'en, Romance of the Three Kingdoms Luo Guanzhong), Renaissance (Shakespeare 38 plays + sonnets, Cervantes Don Quixote, Petrarch, Camões Os Lusíadas, Milton Paradise Lost, Molière, Racine), Enlightenment (Voltaire Candide, Swift Gulliver, Defoe Robinson Crusoe, Goethe Faust/Werther), Romanticism (Wordsworth, Coleridge, Byron, Shelley, Keats, Pushkin, Hugo Les Misérables/Notre-Dame, Schiller, Poe), Realism (Balzac, Flaubert Madame Bovary, Tolstoy War and Peace/Anna Karenina, Dostoevsky Crime and Punishment/Brothers Karamazov, Dickens, Austen, Eliot, Zola, Machado de Assis Dom Casmurro/Memórias Póstumas, Eça de Queirós), Naturalism (Zola, Aluísio Azevedo O Cortiço), Symbolism/Modernism (Baudelaire, Rimbaud, Mallarmé, Rilke, Proust, Kafka Metamorphosis/Trial, Joyce Ulysses, Woolf, Eliot Waste Land, Fitzgerald Gatsby, Hemingway, Faulkner, Borges Ficciones, Pessoa heteronyms, Drummond, Guimarães Rosa, Clarice Lispector), Latin American Boom (García Márquez Cien Años de Soledad, Cortázar, Vargas Llosa, Fuentes, Neruda, Paz, Jorge Amado, Mário de Andrade Macunaíma), Japanese/Chinese/Korean classical and modern (Basho, Kawabata, Mishima, Murakami, Lu Xun, Mo Yan, Han Kang), African literature (Achebe Things Fall Apart, Soyinka, Ngugi, Adichie, Coetzee, Gordimer), Indian literature (Tagore, R.K. Narayan, Arundhati Roy, Rushdie), Arabic (Mahfouz, Darwish, Nizar Qabbani). Canonical works every HS student should know: Iliad, Odyssey, Aeneid, Bible (literary), Qur'an (literary), Divine Comedy, Don Quixote, Hamlet/Macbeth/Romeo&Juliet/King Lear, Paradise Lost, Crime and Punishment, War and Peace, Ulysses, The Great Gatsby, 1984, To Kill a Mockingbird, One Hundred Years of Solitude, Things Fall Apart. PHILOSOPHY: branches (metaphysics, epistemology, ethics, logic, aesthetics, political philosophy, philosophy of mind, philosophy of science, philosophy of language, philosophy of religion), major schools/figures: pre-Socratic (Thales, Heraclitus panta rhei, Parmenides, Pythagoras, Zeno paradoxes, Democritus atoms), Classical (Socrates Socratic method, Plato Forms/Republic/cave allegory, Aristotle Nicomachean Ethics/Politics/Organon/four causes/golden mean), Hellenistic (Stoicism Zeno/Epictetus/Marcus Aurelius, Epicureanism Epicurus, Skepticism Pyrrho, Cynicism Diogenes), Medieval (Augustine Confessions/City of God, Aquinas Summa Theologica/Five Ways, Maimonides, Averroes, Anselm ontological argument), Eastern (Confucius Analects ren/li, Mencius, Laozi Tao Te Ching wu wei, Zhuangzi, Buddha Four Noble Truths/Eightfold Path, Nagarjuna, Shankara, Ramanuja, Dogen), Early Modern (Descartes cogito/Meditations, Spinoza Ethics/substance, Leibniz monads, Hobbes Leviathan, Locke empiricism/tabula rasa, Berkeley idealism, Hume empiricism/is-ought, Rousseau, Kant Critique of Pure Reason/Categorical Imperative, Hegel dialectic/geist, Schopenhauer will), 19th century (Kierkegaard existentialism, Marx/Engels materialism, Nietzsche will to power/eternal return/Übermensch/God is dead, Mill utilitarianism, Bentham, Peirce pragmatism, James), 20th century (Russell, Wittgenstein Tractatus/Investigations, Husserl phenomenology, Heidegger Being and Time/Dasein, Sartre existence precedes essence, de Beauvoir, Camus absurd, Merleau-Ponty, Arendt banality of evil, Popper falsifiability, Kuhn paradigm shifts, Foucault power/knowledge, Derrida deconstruction, Rawls theory of justice/veil of ignorance, Nozick, Habermas, Rorty, Nussbaum, contemporary feminist/postcolonial/critical race theory). ETHICS: consequentialism (utilitarianism act/rule, Bentham/Mill), deontology (Kant, duty, categorical imperative universalizability+humanity-as-end), virtue ethics (Aristotle, eudaimonia, phronesis, MacIntyre), contractarianism (Rawls, Gauthier), care ethics (Gilligan, Noddings), divine command, natural law, moral relativism vs universalism, applied ethics (bioethics: euthanasia, abortion, cloning; business ethics; environmental ethics; animal ethics Singer; AI ethics; just war theory; capital punishment; affirmative action). LOGIC: formal logic (propositional: operators ¬∧∨→↔, truth tables, tautology/contradiction/contingent, equivalence, argument forms modus ponens/modus tollens/hypothetical syllogism/disjunctive syllogism/constructive/destructive dilemma; predicate: quantifiers ∀∃, variables, predicates, domain, universal/existential instantiation and generalization; natural deduction rules; Venn diagrams), syllogistic (Aristotelian: major/minor premise, middle term, distribution, AEIO propositions, figures I-IV, moods, valid forms Barbara/Celarent/Darii/Ferio), informal logic/fallacies (formal: affirming consequent, denying antecedent, undistributed middle; informal: ad hominem, straw man, false dilemma, slippery slope, appeal to authority/emotion/popularity/ignorance/tradition/fear, circular reasoning, post hoc ergo propter hoc, hasty generalization, red herring, equivocation, composition/division, non sequitur, tu quoque, genetic fallacy, Texas sharpshooter, no true Scotsman), inductive reasoning (generalization, analogy, causal inference Mill's methods, Bayesian intro cross-link probability). For each: canonical_id, RPN, symlinks, surface_forms, saudades MASSIVE (every philosopher has locale-specific naming conventions, literary terms differ radically e.g., "saudade" itself is a Portuguese literary concept, German "Weltschmerz", Japanese "mono no aware", Spanish "duende" Lorca — prime saudades ALL).</sub_task_a>
<parameter name="sub_task_b">RELIGIOUS STUDIES (comparative) + ARTS (visual/music/drama/film/dance). RELIGIOUS STUDIES: major world religions: Christianity (Catholic, Orthodox, Protestant Anglican/Lutheran/Calvinist/Baptist/Methodist/Pentecostal; Bible OT+NT, Gospels, Pauline letters, Trinity, Nicene Creed, sacraments, key figures Jesus/Paul/Augustine/Aquinas/Luther/Calvin, liturgy, feasts Christmas/Easter/Pentecost), Islam (Sunni/Shia/Ibadi, Qur'an, Sunnah, Five Pillars Shahada/Salat/Zakat/Sawm/Hajj, Muhammad, Rashidun caliphs, Sunni Four schools Hanafi/Maliki/Shafi'i/Hanbali, Shia Twelver/Ismaili/Zaydi, Sufism Rumi/Ibn Arabi, Eid al-Fitr/Adha, Ramadan), Judaism (Torah, Talmud Babylonian/Jerusalem, Mishnah, Orthodox/Conservative/Reform/Reconstructionist, 613 mitzvot, kosher, Shabbat, High Holidays Rosh Hashanah/Yom Kippur, Pesach, Hanukkah, Sukkot, Tanakh: Torah/Nevi'im/Ketuvim), Hinduism (Vedas Rig/Sama/Yajur/Atharva, Upanishads, Bhagavad Gita, Ramayana, Mahabharata, Trimurti Brahma/Vishnu/Shiva, avatars Rama/Krishna, karma, dharma, samsara, moksha, four yogas Jnana/Karma/Bhakti/Raja, caste concept, Diwali, Holi), Buddhism (Theravada/Mahayana/Vajrayana, Four Noble Truths dukkha, Eightfold Path, Three Marks anicca/anatta/dukkha, Five Precepts, Pali Canon, sutras, Zen, Pure Land, Tibetan, bodhisattva, nirvana, Vesak), Sikhism (Guru Nanak + 9 Gurus, Guru Granth Sahib, Five Ks kesh/kangha/kara/kachera/kirpan, Khalsa, Golden Temple Amritsar), Jainism (Mahavira, ahimsa, five vows, Tirthankaras, Svetambara/Digambara), Taoism (Laozi/Zhuangzi, Tao Te Ching, yin-yang, wu wei, Five Elements, Eight Immortals), Confucianism (Analects, ren/li/xiao/zhong/shu, Five Classics/Four Books, Mencius, Zhu Xi neo-Confucianism), Shinto (kami, Amaterasu, torii, shrines, purity, matsuri), Zoroastrianism (Zarathustra, Avesta, Ahura Mazda/Angra Mainyu, good thoughts/words/deeds), Baháʼí Faith, Indigenous religions (African traditional, Native American, Aboriginal Dreamtime, Shamanism), new religious movements (Mormonism, Jehovah's Witnesses, Scientology, Rastafari), Candomblé/Umbanda/Santería (Afro-Atlantic syncretism). Religious concepts: monotheism/polytheism/pantheism/panentheism/deism/atheism/agnosticism, theology, religious pluralism, secularism, fundamentalism, mysticism, prayer, meditation, ritual, sacrament, pilgrimage, scripture, prophecy, afterlife concepts (heaven/hell/purgatory/reincarnation/limbo/moksha/nirvana), problem of evil, theodicy, religious symbols (cross, crescent, Star of David, Om, dharmachakra, khanda, yin-yang, torii, ankh). VISUAL ARTS: elements (line, shape, form, color RGB primary/CMY secondary/tertiary/complementary/analogous/warm/cool, value, texture, space positive/negative, pattern), principles (balance symmetrical/asymmetrical/radial, contrast, emphasis, movement, pattern, rhythm, unity, variety, proportion, hierarchy, scale), color theory (Itten/Munsell, hue/saturation/brightness, color harmony, psychology of color), techniques (drawing pencil/charcoal/ink, painting oil/watercolor/acrylic/fresco/tempera/gouache/encaustic, printmaking woodcut/etching/lithograph/screenprint, sculpture carving/modeling/assembling/casting/welding, ceramics wheel/hand-building/slip/glaze/firing, photography composition rule of thirds/leading lines/framing, digital art), periods/movements: prehistoric (Lascaux, Venus figurines, megaliths), ancient (Egyptian, Greek Archaic/Classical/Hellenistic, Roman, Chinese Terracotta, Indian Gupta), Byzantine mosaics/icons, Islamic art arabesques/calligraphy/miniature, Medieval Romanesque/Gothic cathedrals/illuminated manuscripts, Renaissance (Early Italian Giotto/Masaccio, High Leonardo Mona Lisa/Last Supper/Vitruvian, Michelangelo David/Sistine, Raphael, Botticelli, Northern van Eyck/Dürer/Bruegel), Mannerism (El Greco, Tintoretto), Baroque (Caravaggio chiaroscuro, Rembrandt, Vermeer, Velázquez Las Meninas, Bernini, Rubens, Aleijadinho Brazilian Baroque), Rococo (Fragonard, Watteau), Neoclassicism (David, Ingres, Canova), Romanticism (Delacroix, Géricault, Goya, Turner, Friedrich), Realism (Courbet, Daumier, Millet), Impressionism (Monet, Renoir, Degas, Manet), Post-Impressionism (Van Gogh Starry Night, Cézanne, Gauguin, Seurat pointillism), Art Nouveau (Mucha, Klimt), Modernism: Fauvism (Matisse), Cubism (Picasso Guernica/Les Demoiselles, Braque), Expressionism (Munch Scream, Kirchner), Futurism (Boccioni), Suprematism/Constructivism (Malevich, Tatlin), Dada (Duchamp Fountain), Surrealism (Dalí, Magritte, Miró, Frida Kahlo), Abstract Expressionism (Pollock, Rothko, de Kooning), Pop Art (Warhol, Lichtenstein, Hamilton), Minimalism, Conceptual, Street Art (Banksy, Basquiat). Architecture: orders Doric/Ionic/Corinthian/Tuscan/Composite, Roman arch/vault/dome (Pantheon), Byzantine (Hagia Sophia), Islamic (Alhambra, Taj Mahal, mosques), Gothic (pointed arch/ribbed vault/flying buttress/rose window, Notre-Dame, Chartres), Renaissance (Brunelleschi Dome, St Peter's), Baroque (Versailles), Neoclassical (US Capitol, White House), Art Nouveau (Gaudí Sagrada Família, Casa Batlló), Modernism (Le Corbusier, Bauhaus Gropius/Mies van der Rohe less is more, Wright Falling Water, Niemeyer Brasília, Aalto), Postmodernism (Gehry Guggenheim Bilbao, Zaha Hadid). MUSIC: elements (pitch, rhythm tempo BPM/meter 4/4/3/4/6/8, duration, dynamics pp/p/mp/mf/f/ff, timbre, texture monophonic/homophonic/polyphonic/heterophonic, form binary AB/ternary ABA/rondo ABACA/sonata/variation/through-composed), notation (staff, treble/bass/alto clef, notes whole/half/quarter/8th/16th, rests, time signature, key signature major/minor, sharps/flats/naturals, accidentals, dynamics/articulation/tempo markings), theory (major/minor scales diatonic, pentatonic, blues, modes Ionian/Dorian/Phrygian/Lydian/Mixolydian/Aeolian/Locrian, intervals m2/M2/m3/M3/P4/tritone/P5/m6/M6/m7/M7/P8, chords triads major/minor/dim/aug, seventh chords maj7/min7/dom7/dim7, inversions, chord progressions I-IV-V-I/ii-V-I/12-bar blues, cadences perfect/imperfect/plagal/deceptive, circle of fifths, transposition, modulation, counterpoint, harmony), genres: classical periods (Medieval Gregorian chant, Renaissance Palestrina/Josquin, Baroque Bach BWV/Handel/Vivaldi, Classical Haydn/Mozart K/Beethoven symphonies/Schubert, Romantic Chopin/Liszt/Wagner/Brahms/Tchaikovsky/Verdi, 20th century Debussy/Ravel/Stravinsky/Schoenberg 12-tone/Bartók/Shostakovich/Copland), jazz (Dixieland, swing Ellington/Basie, bebop Parker/Gillespie, cool Davis, hard bop, free jazz Coltrane, fusion), blues (12-bar, Delta, Chicago, B.B. King), rock (early Elvis/Chuck Berry, Beatles/Stones, psychedelic, hard/heavy metal, punk, new wave, grunge, alt, indie), folk (American Dylan/Baez, world), country, R&B/soul Motown/Stax, funk, disco, hip-hop (old school, golden age, gangsta, conscious, Southern, trap), electronic (house, techno, trance, drum&bass, dubstep, EDM), world music (Indian classical raga/tala, Chinese traditional, Japanese gagaku/shamisen, Latin samba/bossa nova Tom Jobim/tango/salsa/bolero/mariachi, African drumming/highlife, Celtic, Arabic maqam, Brazilian MPB/forró/choro/axé/funk carioca, Portuguese fado Amália Rodrigues), instruments (strings violin family, guitar, harp, piano; woodwind flute/clarinet/oboe/bassoon/saxophone; brass trumpet/trombone/horn/tuba; percussion timpani/xylophone/drums; keyboards organ/harpsichord/synth; voice soprano/mezzo/alto/tenor/baritone/bass). DRAMA/THEATER: genres (tragedy, comedy, tragicomedy, melodrama, farce, satire, absurdist, musical theater), historical theater (Greek Aeschylus/Sophocles/Euripides tragedy + Aristophanes comedy, chorus, mask, theatron; Roman Plautus/Terence/Seneca; Medieval mystery/miracle/morality plays; commedia dell'arte; Elizabethan Shakespeare/Marlowe/Jonson; French classical Molière/Racine/Corneille; Restoration; 19th century melodrama/well-made play; Modern Ibsen A Doll's House, Chekhov, Strindberg, Shaw, O'Neill, Miller Death of a Salesman, Williams Streetcar, Beckett Waiting for Godot, Brecht epic theater, Ionesco, Pinter; Brazilian Nelson Rodrigues, Ariano Suassuna Auto da Compadecida; Japanese Noh/Kabuki/Bunraku; Chinese Peking opera/kunqu; Indian Sanskrit drama/Kathakali); acting techniques (Stanislavski method, Meisner, Chekhov technique, Brechtian, Viewpoints, Suzuki), stagecraft (set, lighting, costume, sound, props, makeup, directing, dramaturgy). FILM: language (shot types ELS/LS/MS/CU/ECU, angles low/high/Dutch/bird's eye, movements pan/tilt/track/dolly/crane/Steadicam/zoom, cuts jump/match/cross/montage, mise-en-scène), history (Lumière/Méliès silent, Chaplin/Keaton/Murnau/Eisenstein Soviet montage Battleship Potemkin, sound era 1927 Jazz Singer, Hollywood Golden Age, Italian neorealism De Sica/Rossellini, French New Wave Godard/Truffaut, Japanese Kurosawa/Ozu/Mizoguchi, New Hollywood Coppola Godfather/Scorsese/Spielberg/Lucas, Cinema Novo Glauber Rocha, Dogme 95, contemporary auteurs). DANCE: ballet (positions, pliés, tendus, classical Nutcracker/Swan Lake/Giselle, Balanchine, contemporary), modern dance (Isadora Duncan, Martha Graham, Cunningham), jazz dance, tap, ballroom (waltz, tango, foxtrot, cha-cha, samba, rumba), folk/cultural (flamenco, Irish, Bharatanatyam, Kathak, samba, hula, Cossack, belly dance, capoeira). For each: canonical_id, RPN, symlinks (literary devices cross-link to grammar cluster, music theory cross-link to math frequencies/ratios, arts to history events), surface_forms, saudades MASSIVE (pt "saudade" itself, ja "mono no aware" 物の哀れ / "wabi-sabi" 侘寂, de "Weltschmerz" / "Sehnsucht" / "Gemütlichkeit", es "duende", ru "toska" тоска, fr "flâneur" / "joie de vivre", it "sprezzatura", etc. — flag all).</sub_task_b>
<parameter name="context">K3D Phase 7.A.2 HS world-curriculum humanities seed. Opcode palette strict. Canonical ID: work_<snake>, author_<snake>, philosopher_<snake>, movement_<snake>, religion_<snake>, deity_<snake>, scripture_<snake>, doctrine_<snake>, artwork_<snake>, composer_<snake>, genre_<snake>, period_<snake>, device_<snake>, concept_<snake>. Cross-cluster symlinks: literary devices↔grammar cluster, music↔math (ratios/frequencies/Pythagorean tuning), arts↔history events, philosophy↔logic↔reasoning cluster. Saudades: THIS CLUSTER HAS THE HIGHEST DENSITY OF UNTRANSLATABLE TERMS in the entire curriculum. Daniel specifically wants saudades ("saudade") itself as a prime example — flag the whole list of language-specific literary/philosophical/aesthetic concepts (saudade pt, saudadezinha pt, mono no aware ja, wabi-sabi ja, iki ja, komorebi ja, tsundoku ja, ikigai ja, Weltschmerz de, Sehnsucht de, Gemütlichkeit de, Fernweh de, Schadenfreude de, Waldeinsamkeit de, duende es, sobremesa es, toska ru, poshlost ru, flâneur fr, savoir-faire fr, joie de vivre fr, sprezzatura it, dolce far niente it, and literary/music/philosophy terms). 9-language surface_forms.</parameter>
<parameter name="web_queries">world literature canonical works high school, literary devices complete list, AP literature syllabus, major philosophers timeline, ethics branches consequentialism deontology virtue, comparative religion five major world religions, art movements complete timeline, music theory high school, world music genres, film studies high school

**K3D High-School Humanities Seed — Sub-Agent B Output**
**Domain:** Religious Studies + Arts (Visual / Music / Drama / Film / Dance)  
**Canonical ID Prefixes:** `religion_`, `scripture_`, `doctrine_`, `deity_`, `artwork_`, `movement_`, `genre_`, `composer_`, `device_`, `concept_`, `technique_`, `form_`

---

## 1. RELIGIOUS STUDIES / COMPARATIVE RELIGION

### 1.1 Major World Religions (Institutional Nodes)

| canonical_id | is_a | RPN Sketch (9-language surface_forms below) | Symlinks | Saudades |
|--------------|------|---------------------------------------------|----------|----------|
| **religion_christianity** | `[religion_abrahamic, religion_monotheistic]` | `deity_yahweh RECALL deity_jesus RECALL doctrine_trinity RECALL scripture_bible RECALL liturgy_eucharist RECALL TPACK STORE` | `<-> religion_judaism (parent)`<br>`<-> deity_jesus`<br>`<-> scripture_bible`<br>`<-> history_event_reformation` | no |
| **religion_islam** | `[religion_abrahamic, religion_monotheistic]` | `deity_allah RECALL prophet_muhammad RECALL scripture_quran RECALL pillar_shahada RECALL mosque_kibla RECALL TPACK STORE` | `<-> religion_christity (sibling)`<br>`<-> scripture_quran`<br>`<-> concept_sharia` | no |
| **religion_judaism** | `[religion_abrahamic, religion_ethnic]` | `deity_yahweh RECALL covenant_abraham RECALL scripture_tanakh RECALL torah RECALL mitzvot_613 RECALL TPACK STORE` | `<-> religion_islam (sibling)`<br>`<-> scripture_talmud`<br>`<-> history_event_exodus` | yes (hebr: *davka*, *tikkun olam*) |
| **religion_hinduism** | `[religion_dharmic, religion_polytheistic_pantheistic]` | `trimurti_brahma_vishnu_shiva RECALL concept_dharma RECALL concept_karma RECALL scripture_vedas RECALL TPACK STORE` | `<-> concept_reincarnation`<br>`<-> concept_moksha`<br>`<-> art_form_kathakali` | yes (sanskr: *darshan*, *rasa*) |
| **religion_buddhism** | `[religion_dharmic, religion_non_theistic]` | `buddha_siddhartha RECALL truth_dukkha RECALL path_eightfold RECALL concept_nirvana RECALL TPACK STORE` | `<-> concept_samsara`<br>`<-> scripture_pali_canon`<br>`<-> art_form_mandala` | yes (pal: *dukkha* → ja: 苦) |
| **religion_sikhism** | `[religion_dharmic, religion_monotheistic]` | `guru_nanak RECALL scripture_guru_granth RECALL five_ks RECALL khalsa RECALL TPACK STORE` | `<-> religion_hinduism (neighbor)`<br>`<-> concept_langar` | yes (pa: *chardi kala*) |
| **religion_taoism** | `[religion_philosophical, east_asian]` | `laozi RECALL text_daodejing RECALL concept_wu_wei RECALL yin_yang RECALL TPACK STORE` | `<-> philosophy_daoism`<br>`<-> concept_qi` | yes (zh: 道, 无为, 阴阳) |
| **religion_confucianism** | `[religion_philosophical, east_asian]` | `confucius RECALL analects RECALL concept_ren RECALL concept_li RECALL five_relationships RECALL TPACK STORE` | `<-> religion_taoism (counterpart)`<br>`<-> concept_xiao` | yes (zh: 仁, 孝, 礼) |
| **religion_shinto** | `[religion_indigenous, japanese]` | `kami_spirits RECALL shrine_torii RECALL concept_purity RECALL matsuri RECALL TPACK STORE` | `<-> concept_amaterasu`<br>`<-> architecture_torii`

────────────────────────────────────────────────────────────────

## Synthesis

The user wants me to act as the Synthesis Agent in a Kimi Agent Swarm. I need to merge the findings from Sub-Agent A and Sub-Agent B into a unified, actionable answer.

Looking at the inputs:
- Sub-Agent A focused on World Literature + Philosophy/Ethics/Logic
- Sub-Agent B focused on Religious Studies + Arts (Visual/Music/Drama/Film/Dance)

The task is to produce a K3D meaning-star plan for HIGH-SCHOOL HUMANITIES covering all these domains.

Key requirements from the context:
1. Each meaning-star needs: canonical_id, is_a, RPN sketch using ONLY specific opcodes (STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK)
2. 9-language surface_forms (en/pt/es/fr/de/it/ja/zh/ru)
3. Bidirectional symlinks
4. Flag saudades candidates heavily (untranslatable terms)
5. Cross-cluster symlinks specified: literary devices↔grammar, music↔math, arts↔history, philosophy↔logic↔reasoning

Conflicts to resolve:
- Sub-Agent A used some RPN sketches but Sub-Agent B didn't complete the table format fully for its section.
- I need to unify the format across all domains.
- Ensure the opcode palette is strictly followed (only the allowed opcodes).
- Ensure all saudades are flagged (the context mentions this cluster has the highest density of untranslatable terms).

Insights to add:
- Cross-domain linkages (e.g., how religious concepts link to literature, how music theory links to math)
- Ensure the RPN sketches are valid using only the allowed opcodes
- Make sure all 9 languages are represented in surface_forms
- Ensure canonical IDs follow the snake_case pattern specified

Structure:
The output should be a Codex spec or implementation guide. This means structured, technical, actionable.

Plan:
1. Introduction/Meta-commentary on the merge
2. Unified Schema Definition (canonical ID patterns, opcode constraints)
3. Domain Sections:
   - Literature (Genres, Forms, Devices, Movements)
   - Philosophy/Ethics/Logic (Branches, Schools, Concepts)
   - Religious Studies (Religions, Doctrines, Practices)
   - Arts (Visual, Music, Drama, Film, Dance)
4. Cross-Cluster Symlink Matrix
5. Saudades Master List (since this is flagged as high density)
6. Implementation Notes

For each entry, I need:
- canonical_id (work_, author_, philosopher_, movement_, religion_, deity_, scripture_, doctrine_, artwork_, composer_, genre_, period_, device_, concept_)
- is_a (parent class)
- RPN: valid stack operations using only allowed opcodes
- surface_forms: 9 languages
- symlinks: bidirectional references
- saudades: boolean flag

Let me draft specific entries that merge and complete both agents' work:

Literature section:
- Merge the genre/form entries from Agent A
- Ensure saudades like "saudade", "Weltschmerz", "mono no aware", "duende" are properly flagged and defined
- Add missing canonical works (Agent A listed many but I should ensure the RPN is valid)

Philosophy section:
- Need to cover branches and figures
- Logic nodes need to link to reasoning cluster

Religion section:
- From Agent B, formalize the table structure that was incomplete
- Add saudades like "dharma", "karma", "dukkha", "wu wei", "ren"

Arts section:
- Visual arts movements and elements
- Music theory (linking to math)
- Film terminology
- Dance forms

Critical constraints check:
- Opcodes allowed: STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK
- No arithmetic literals in RPN except as stack operations? Actually, the RPN should use these opcodes to manipulate semantic tokens.
- Format: Each RPN sketch should show the stack-based operations using these specific opcodes.

Let me construct valid RPN examples:
- For a complex concept: `base_concept STORE attribute1 RECALL attribute2 RECALL TADD TQUANT` (combining attributes then quantifying)
- For branching: `condition STORE OP_BRANCH true_path false_path`
- For packing: `item1 STORE item2 STORE item3 STORE TPACK` (creates tuple)
- For unpacking: `tuple RECALL TUNPACK`
- For comparison: `a RECALL b RECALL TCOMP` (returns -1, 0, 1)
- For multiplication (intersection): `set_a RECALL set_b RECALL TMUL`
- For addition (union): `set_a RECALL set_b RECALL TADD`
- For negation: `concept RECALL TNOT`
- For galaxy lookup: `query_term GALAXY_LOOKUP`

Saudades to include (building from context and agents):
Portuguese: saudade, saudadezinha, fado (related), desenrascanço
Japanese: mono no aware, wabi-sabi, iki, komorebi, tsundoku, ikigai, yūgen, ma
German: Weltschmerz, Sehnsucht, Gemütlichkeit, Fernweh, Schadenfreude, Waldeinsamkeit, Bildung, Gestalt, Sturm und Drang
Spanish: duende, sobremesa, quixotic (from Don Quixote), querencia
Russian: toska, poshlost, zapoy
French: flâneur, savoir-faire, joie de vivre, nostalgie, terroir
Italian: sprezzatura, dolce far niente, commedia dell'arte (as concept)
Arabic: tarab, halal/haram (though these are doctrinal), yaqeen, sabr
Hebrew: davka, tikkun olam, tsimtsum
Chinese: yuanfen, guanxi, wu wei, li, ren, xiao, qi, yin-yang
Sanskrit: darshan, rasa, moksha, samsara, karma, dharma

Now constructing the actual entries...

For Literature - Canonical Works:
- work_iliad: epic_poetry STORE greek_heroic RECALL oral_tradition RECALL TADD TQUANT
- work_divine_comedy: journey_store afterlife RECALL terza_rima RECALL allegory TADD TPACK

For Philosophy:
- philosopher_plato: socratic_method STORE forms_theory RECALL cave_allegory RECALL TADD TQUANT
- concept_categorical_imperative: universalizability STORE humanity_as_end RECALL duty RECALL TADD TQUANT

For Logic:
- logic_modus_ponens: premise_major STORE premise_minor RECALL implication RECALL TADD TQUANT (simplified)

Actually, looking at the RPN more carefully, it seems the tokens between opcodes are semantic labels, not variables. So: `token1 STORE token2 RECALL TADD` etc.

Let me ensure the structure is consistent with Agent A's example:
`epic_poetry STORE tradition_oral RECALL heroic_narrative TADD TQUANT`

This seems to be: [concept] STORE [attribute] RECALL [attribute2] RECALL TADD TQUANT
Wait, that doesn't make sense stack-wise.
If we push epic_poetry, then STORE stores it somewhere? Or is STORE `value STORE key`?
Actually in stack machines: `value key STORE` stores value at key. `key RECALL` pushes value.
But Agent A wrote: `epic_poetry STORE tradition_oral RECALL...`

This looks like: 
1. push epic_poetry
2. STORE (pops value and key?) - but tradition_oral is next?
Actually standard Forth-like RPN:
- `value variable STORE` stores value in variable
- `variable RECALL` pushes value

So Agent A's: `epic_poetry STORE tradition_oral RECALL heroic_narrative TADD TQUANT`
Would mean:
- push epic_poetry
- push tradition_oral
- STORE (stores epic_poetry in tradition_oral? No, that reverses it)
Wait, it should be: `epic_poetry tradition_oral STORE` to store epic_poetry in tradition_oral.

But Agent A wrote: `epic_poetry STORE tradition_oral RECALL...`
Maybe their convention is: STORE takes the value below the key? No, that's not standard.

Actually, looking carefully at Agent A:
`epic_poetry STORE tradition_oral RECALL heroic_narrative TADD TQUANT`

Perhaps the syntax is: tokens are separated by spaces, and STORE/RECALL are operators.
If so:
- push epic_poetry
- STORE (???)
This is ambiguous.

Alternative interpretation: The RPN is describing the structure semantically, not as a running program. But the instruction says "RPN sketch using ONLY opcodes".

Let me look at the allowed opcodes again: STORE, RECALL, GALAXY_LOOKUP, OP_BRANCH, TQUANT, TCOMP, TADD, TMUL, TNOT, TPACK, TUNPACK.

These look like:
- STORE: store top of stack to memory
- RECALL: push from memory to stack
- GALAXY_LOOKUP: lookup in knowledge graph
- OP_BRANCH: conditional branch
- TQUANT: quantify/transform?
- TCOMP: compare
- TADD: add/merge
- TMUL: multiply/intersect
- TNOT: negate
- TPACK: pack into tuple
- TUNPACK: unpack tuple

Given Agent A's example: `epic_poetry STORE tradition_oral RECALL heroic_narrative TADD TQUANT`
Maybe it means:
- Store epic_poetry in tradition_oral slot, then recall heroic_narrative, add them, quantify?
This is confusing.

Better interpretation: The RPN sketches are using the opcodes as functions that operate on implicit stack. The tokens like `epic_poetry` are conceptual atoms.

Valid RPN sequence:
1. Push epic_poetry
2. Push tradition_oral
3. STORE (pops both, stores epic_poetry at address tradition_oral)
4. Push heroic_narrative
5. RECALL (pushes content of heroic_narrative?)
No, RECALL usually takes an address.

Let me reconsider. In some educational RPN systems for semantic networks:
- `A B STORE` means store A in B (B is the container)
- `A RECALL` means recall A

So Agent A's string: `epic_poetry STORE tradition_oral RECALL heroic_narrative TADD TQUANT`
If parsed as: ((epic_poetry STORE tradition_oral) RECALL heroic_narrative TADD) TQUANT
This would mean:
- Store epic_poetry in tradition_oral (but then what is left on stack?)
- Then RECALL that stored value?
- Then TADD with heroic_narrative?
- Then TQUANT?

Actually, I think the format is:
[subject] STORE [context] RECALL [predicate] TADD [quantifier] TQUANT
Or similar.

To avoid confusion, I will follow Agent A's pattern exactly:
`concept STORE attribute RECALL attribute2 RECALL TADD TQUANT`
This seems to be the established pattern in the partial output.

So for new entries, I'll use:
`primary_concept STORE secondary_aspect RECALL tertiary_aspect RECALL TADD TQUANT`
Or for packing multiple attributes:
`attr1 STORE attr2 STORE attr3 STORE TPACK`

And for logic operations (intersection of concepts):
`concept_a RECALL concept_b RECALL TMUL`

Union:
`concept_a RECALL concept_b RECALL TADD`

Negation:
`concept RECALL TNOT`

Comparison:
`a RECALL b RECALL TCOMP`

Branching:
`condition STORE OP_BRANCH path_a path_b`

Galaxy lookup:
`term GALAXY_LOOKUP`

Now, merging the content:

Sub-Agent A provided:
- Literary genres & forms table (partial)
- Literary devices table (partial)

Sub-Agent B provided:
- Religious studies table (partial, header only)
- Mentioned arts domains but didn't provide table

So I need to complete:
- Full literature table (completing Agent A)
- Philosophy/Ethics/Logic table (Agent A mentioned but didn't fully table-ize, only gave headers)
- Religious Studies (completing Agent B's header)
- Arts (Visual, Music, Drama, Film, Dance) - Agent B mentioned but didn't table-ize

Also need to ensure:
- All 9 languages in surface_forms
- Bidirectional symlinks (e.g., `<-> concept_x`)
- Saudades flag (TRUE/FALSE or yes/no, but Agent A used TRUE/FALSE)

Canonical ID prefixes to use:
work_, author_, philosopher_, movement_, religion_, deity_, scripture_, doctrine_, artwork_, composer_, genre_, period_, device_, concept_

is_a values should be classes like: literary_genre, literary_form, poetic_form, figurative_language, aesthetic_concept, religion_abrahamic, religion_dharmic, etc.

Given the massive scope, I should provide representative examples for each category rather than exhaustively listing every single work (which would be thousands of entries). The prompt asks for a "plan" and "Codex spec", so a structured template with key examples is appropriate, noting that full implementation would expand these.

However, the agents were asked to produce the plan for the specific listed items. I should try to cover the major ones mentioned in the task description.

Let me structure the Synthesis:

1. **Unified Schema Specification**
   - Opcode semantics
   - Surface form encoding (9-lang)
   - Symlink notation
   - Saudades taxonomy

2. **Literature Cluster** (World Literature)
   - Genres (epic, lyric, drama, etc.)
   - Forms (sonnet, haiku, etc.)
   - Devices (metaphor, saudade, etc.)
   - Movements (Classical, Medieval, Renaissance, etc.)
   - Canonical Works (Iliad, Odyssey, etc.)

3. **Philosophy/Ethics/Logic Cluster**
   - Branches (metaphysics, epistemology, etc.)
   - Historical Schools (Pre-Socratic, Classical, etc.)
   - Ethics Systems (consequentialism, deontology, etc.)
   - Logic Constructs (modus ponens, fallacies, etc.)

4. **Religious Studies Cluster**
   - Religions (Christianity, Islam, etc.)
   - Doctrines (Trinity, Five Pillars, etc.)
   - Scriptures (Bible, Quran, etc.)
   - Concepts (Karma, Dharma, etc.)

5. **Arts Cluster**
   - Visual Arts (movements, elements, techniques)
   - Music (theory, genres, forms)
   - Drama/Theater (genres, techniques)
   - Film (language, movements)
   - Dance (forms)

6. **Cross-Cluster Symlink Matrix** (showing the bidirectional links)

7. **Saudades Master Registry** (alphabetical by language)

8. **Implementation Notes** (for Codex generation)

Let me draft specific high-quality entries:

**Literature Examples:**

| canonical_id | is_a | RPN Sketch | surface_forms | symlinks | saudades |
|--------------|------|------------|---------------|----------|----------|
| device_saudade | aesthetic_concept | `absence STORE presence RECALL longing RECALL melancholy RECALL TADD TQUANT` | en:saudade; pt:saudade; es:saudade/añoranza; fr:saudade; de:Saudade/Sehnsucht; it:saudade; ja:サウダージ; zh:萨乌达德; ru:саудаде | `<-> genre_fado`<br>`<-> concept_melancholy`<br>`<-> device_weltschmerz` | TRUE |
| form_haiku | poetic_form | `syllables_5_7_5 STORE kigo RECALL nature_impermanence RECALL TADD TQUANT` | en:haiku; pt:haicai; es:haiku; fr:haïku; de:Haiku; it:haiku; ja:俳句; zh:俳句; ru:хайку | `<-> concept_kigo`<br>`<-> author_basho`<br>`<-> device_mono_no_aware` | TRUE |
| device_mono_no_aware | aesthetic_concept | `impermanence STORE pathos RECALL beauty_transience RECALL TADD TQUANT` | en:mono no aware; pt:mono no aware; es:mono no aware; fr:pathétique des choses; de:Mono no aware; it:mono no aware; ja:物の哀れ; zh:物哀; ru:моно но аварэ | `<-> form_haiku`<br>`<-> author_murasaki`<br>`<-> concept_wabi_sabi` | TRUE |
| device_weltschmerz | aesthetic_concept | `world_sorrow STORE pessimism RECALL romanticism_origin RECALL TADD TQUANT` | en:Weltschmerz; pt:Weltschmerz; es:Weltschmerz; fr:mal du siècle; de:Weltschmerz; it:Weltschmerz; ja:ヴェルトシュメルツ; zh:厌世; ru:мировая боль | `<-> movement_romanticism`<br>`<-> author_heine` | TRUE |
| device_duende | aesthetic_concept | `soul_darkness STORE artistic_daemon RECALL flamenco_origin RECALL TADD TQUANT` | en:duende; pt:duende; es:duende; fr:duende; de:Duende; it:duende; ja:ドゥエンデ; zh:灵魔; ru:дуэндэ | `<-> movement_symbolism`<br>`<-> music_flamenco` | TRUE |

**Philosophy Examples:**

| canonical_id | is_a | RPN Sketch | surface_forms | symlinks | saudades |
|--------------|------|------------|---------------|----------|----------|
| philosopher_kant | philosopher | `criticism_pure_reason STORE categorical_imperative RECALL enlightenment_project RECALL TADD TQUANT` | en:Immanuel Kant; pt:Immanuel Kant; es:Immanuel Kant; fr:Emmanuel Kant; de:Immanuel Kant; it:Immanuel Kant; ja:カント; zh:康德; ru:Кант | `<-> concept_categorical_imperative`<br>`<-> ethics_deontology`<br>`<-> logic_reasoning` | FALSE |
| concept_categorical_imperative | ethical_principle | `universalizability STORE humanity_as_end RECALL autonomy RECALL TADD TQUANT` | en:categorical imperative; pt:imperativo categórico;
