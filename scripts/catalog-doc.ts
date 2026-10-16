/** Writes docs/MODELS.md from the model catalog: npx tsx scripts/catalog-doc.ts */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { FAMILIES, modelsOf } from '../src/models/catalog';

const lines = [
  '# Models',
  '',
  'Generated from `src/models/catalog.ts` by `scripts/catalog-doc.ts`. Do not edit by hand.',
  '',
];
for (const f of FAMILIES) {
  lines.push(`## ${f.name}`, '', f.summary, '');
  lines.push(
    '| Model | Scale | Training here | Pretrained weights | Reference |',
    '|---|---|---|---|---|',
  );
  for (const m of modelsOf(f.id)) {
    const ref = m.references[0];
    const first = ref?.authors.split(/,| and /)[0].trim();
    const count = ref ? ref.authors.split(/,| and /).filter((a) => a.trim()).length : 0;
    const who = count > 2 ? `${first} et al.` : ref?.authors.trim();
    const refText = ref
      ? ref.url
        ? `[${who} ${ref.year}](${ref.url})`
        : `${who} ${ref.year}`
      : '';
    lines.push(
      `| [${m.name}](../src/models/${f.id}) \`${m.id}\` | ${m.fidelity === 'exact' ? 'published size' : m.fidelity === 'equations' ? 'published equations' : 'reduced'} | ${m.training.support} | ${m.checkpoints?.length ? 'yes' : '—'} | ${refText} |`,
    );
  }
  lines.push('');
  for (const m of modelsOf(f.id)) {
    lines.push(`**${m.name}.** ${m.summary}${m.scaleNote ? ` *Scale:* ${m.scaleNote}` : ''}`, '');
    for (const l of m.limitations) lines.push(`- ${l}`);
    lines.push('');
  }
}
writeFileSync(join(import.meta.dirname, '..', 'docs', 'MODELS.md'), lines.join('\n'));
console.log('wrote docs/MODELS.md');
