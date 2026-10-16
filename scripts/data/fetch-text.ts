/**
 * Downloads Andrej Karpathy's tiny-shakespeare file (a concatenation of
 * passages from Shakespeare's plays; the text is in the public domain) into
 * scripts/.data/text and writes the subset shipped with the site to
 * public/data/text/shakespeare.txt: the first paragraphs of the file, cut at
 * a blank line, at most MAX_BYTES bytes.
 *
 * Usage: npx tsx scripts/data/fetch-text.ts
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { SHAKESPEARE_VOCAB } from '../../src/models/data/text';

const URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const CACHE = join(import.meta.dirname, '..', '.data', 'text');
const OUT = join(import.meta.dirname, '..', '..', 'public', 'data', 'text');
const MAX_BYTES = 300_000;

export async function loadTinyShakespeare(): Promise<string> {
  mkdirSync(CACHE, { recursive: true });
  const path = join(CACHE, 'tinyshakespeare.txt');
  if (!existsSync(path)) {
    const res = await fetch(URL);
    if (!res.ok) throw new Error(`download tiny-shakespeare: HTTP ${res.status}`);
    writeFileSync(path, new Uint8Array(await res.arrayBuffer()));
  }
  return readFileSync(path, 'utf8');
}

async function main() {
  const full = await loadTinyShakespeare();
  const vocab = new Set(SHAKESPEARE_VOCAB);
  const unknown = [...new Set(full)].filter((c) => !vocab.has(c));
  if (unknown.length)
    throw new Error(`characters outside the fixed vocabulary: ${JSON.stringify(unknown)}`);
  // The file is ASCII, so characters and bytes coincide.
  let end = full.lastIndexOf('\n\n', MAX_BYTES);
  if (end < 0) end = MAX_BYTES;
  const subset = full.slice(0, end + 1);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, 'shakespeare.txt'), subset);
  console.log(
    `full file ${full.length} characters; wrote ${subset.length} characters to ${OUT}/shakespeare.txt`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
