export interface ContentEntry {
  star_id: string;
  meaning_class: string;
  domain: string;
  meaning_rpn: string;
  behavior_rpn: string | null;
  surface_forms: Record<string, { word_ref: string; char_refs: string[] }>;
  taxonomy_refs: string[];
  grammar_refs: string[];
  component_refs: string[];
  visual_rpn?: string;
}

export interface BookContent {
  entries: ContentEntry[];
}

export interface HouseContent {
  version: number;
  books: Record<string, BookContent>;
  concepts: Record<string, ContentEntry>;
}

let houseContent: HouseContent | null = null;

export async function loadHouseContent(url: string): Promise<HouseContent> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  houseContent = await response.json() as HouseContent;
  return houseContent;
}

export function clearHouseContent(): void {
  houseContent = null;
}

export function getBookContent(galaxyRef: string): BookContent | null {
  return houseContent?.books?.[galaxyRef] || null;
}

export function getConceptEntry(conceptId: string): ContentEntry | null {
  return houseContent?.concepts?.[conceptId] || null;
}

export function resolveRef(ref: string): ContentEntry | null {
  const concept = getConceptEntry(ref);
  if (concept) return concept;
  if (!houseContent) return null;
  for (const book of Object.values(houseContent.books)) {
    const entry = book.entries.find((candidate) => candidate.star_id === ref);
    if (entry) return entry;
  }
  return null;
}

export function getLoadedContent(): HouseContent | null {
  return houseContent;
}

