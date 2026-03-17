import type { HouseNode } from '../loadHouseScene';

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
