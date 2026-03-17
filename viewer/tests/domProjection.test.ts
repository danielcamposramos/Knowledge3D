import { buildNodeDomProgram } from '../src/behavior/domProjection';
import { rpnToDOM } from '../src/rpn';

describe('DOM projection from ContentPage', () => {
  it('projects a content page to DOM', () => {
    const page = {
      title: 'Mathematics Primer',
      sections: [
        { heading: 'Identity', lines: ['Star ID: book_math', 'Class: book'] },
        { heading: 'References', lines: ['concept_mathematics'] },
      ],
    };
    const program = buildNodeDomProgram(page);
    expect(program).toBeTruthy();
    const root = rpnToDOM(program!);
    expect(root.querySelector('h2')?.textContent).toBe('Mathematics Primer');
    const sections = root.querySelectorAll('section');
    expect(sections.length).toBe(2);
    expect(sections[0]?.querySelector('h3')?.textContent).toBe('Identity');
  });

  it('returns null for null page', () => {
    expect(buildNodeDomProgram(null)).toBeNull();
  });
});
