import type { HouseNode } from '../loadHouseScene';
import type { BookContent } from '../contentLoader';

export interface ContentSection {
  heading: string;
  lines: string[];
}

export interface ContentPage {
  title: string;
  sections: ContentSection[];
}

export interface ContentPayload {
  node?: HouseNode;
  title?: string;
  galaxyRef?: string;
  taxonomyRefs?: string[];
  behaviorRpn?: string;
  visualRpn?: string;
  meaningClass?: string;
  domain?: string;
}

export function renderNodeContent(node: HouseNode): ContentPage {
  return renderContentPayload({ node });
}

export function renderBookContent(
  galaxyRef: string,
  book: BookContent,
  node: HouseNode,
): ContentPage {
  const title = node.surfaceForms.en?.word_ref || node.surfaceForms.pt?.word_ref || galaxyRef;
  const sections: ContentSection[] = [];
  const chapters = book.entries.filter((entry) => entry.meaning_class === 'chapter');
  const otherEntries = book.entries.filter((entry) => entry.meaning_class !== 'chapter');

  for (const chapter of chapters) {
    const chapterTitle = chapter.surface_forms.en?.word_ref || chapter.star_id;
    const lines: string[] = [];
    for (const componentRef of chapter.component_refs) {
      const child = book.entries.find((entry) => entry.star_id === componentRef);
      if (!child) continue;
      const childTitle = child.surface_forms.en?.word_ref || child.star_id;
      lines.push(`${child.meaning_class === 'section' ? '§' : '⁋'} ${childTitle}`);
      const refs = child.taxonomy_refs.slice(0, 4).join(', ');
      if (refs) lines.push(`  refs: ${refs}`);
      if (child.grammar_refs.length) {
        lines.push(`  rules: ${child.grammar_refs.join(', ')}`);
      }
    }
    sections.push({ heading: chapterTitle, lines });
  }

  const referencedIds = new Set(chapters.flatMap((chapter) => chapter.component_refs));
  const unreferenced = otherEntries.filter((entry) => !referencedIds.has(entry.star_id));
  if (unreferenced.length) {
    sections.push({
      heading: 'Additional Entries',
      lines: unreferenced.map((entry) => {
        const name = entry.surface_forms.en?.word_ref || entry.star_id;
        return `${entry.meaning_class}: ${name}`;
      }),
    });
  }

  sections.push({
    heading: 'Book Info',
    lines: [
      `Galaxy: ${galaxyRef}`,
      `Entries: ${book.entries.length}`,
      `Chapters: ${chapters.length}`,
      `References: ${new Set(book.entries.flatMap((entry) => entry.taxonomy_refs)).size} unique`,
    ],
  });

  return { title, sections };
}

export function renderContentPayload(payload: ContentPayload): ContentPage {
  const node = payload.node;
  const title = payload.title || node?.surfaceForms.en?.word_ref || node?.surfaceForms.pt?.word_ref || node?.starId || 'House Content';
  const sections: ContentSection[] = [];

  sections.push({
    heading: 'Identity',
    lines: [
      `Star ID: ${node?.starId || '—'}`,
      `Class: ${payload.meaningClass || node?.meaningClass || '—'}`,
      `Domain: ${payload.domain || node?.domain || '—'}`,
      `Room: ${node?.houseRoom || '—'}`,
    ],
  });

  const formLines: string[] = [];
  if (node) {
    for (const [lang, form] of Object.entries(node.surfaceForms)) {
      formLines.push(`${lang}: ${form.word_ref}`);
    }
  }
  if (formLines.length) {
    sections.push({ heading: 'Names', lines: formLines });
  }

  const refs = payload.taxonomyRefs || node?.taxonomyRefs || [];
  if (refs.length) {
    sections.push({
      heading: 'References',
      lines: refs.map((ref) => `→ ${ref}`),
    });
  }

  const galaxyRef = payload.galaxyRef || node?.galaxyRef || '';
  if (galaxyRef) {
    sections.push({
      heading: 'Content Galaxy',
      lines: [`Load: ${galaxyRef}`],
    });
  }

  const componentRefs = node?.componentRefs || [];
  if (componentRefs.length) {
    sections.push({
      heading: 'Components',
      lines: componentRefs.map((ref) => `• ${ref}`),
    });
  }

  const visualRpn = payload.visualRpn || node?.visualRpn || '';
  if (visualRpn) {
    sections.push({
      heading: 'Visual Program',
      lines: [visualRpn],
    });
  }

  sections.push({
    heading: 'Behavior Program',
    lines: [payload.behaviorRpn || node?.behaviorRpn || '—'],
  });

  return { title, sections };
}
