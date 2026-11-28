// Switch an installed agentic-kit project to "private mode": planning files live in a local-only folder
// (default .planning) and, with .claude/ and CLAUDE.md, never reach git. Public docs stay in docs/.
//   node scripts/agentic/private-mode.ts [--project <dir>] [--dir .planning] [--dry-run]
// Idempotent. Backs up what it touches to logs/private-mode-backup-<ts>.tgz first.
import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";

// Planning files that move out of docs/. Everything else in docs/ (ADRs, ai-design, slo, threat-model,
// model-card, runbook, perf, ...) is a public deliverable and stays.
const PLANNING = [
  "spec.md",
  "status.md",
  "backlog.md",
  "tickets",
  "release",
  "implementation_plan.md",
  "quality-rubric.md",
  "evidence-map.md",
  "spec.lean-v0.md",
];
const PRIVATE_PATHS = (dir: string): string[] => [dir, ".claude", "CLAUDE.md", "logs"];

const arg = (n: string): string | undefined => {
  const i = process.argv.indexOf(`--${n}`);
  return i > -1 ? process.argv[i + 1] : undefined;
};
const dry = process.argv.includes("--dry-run");
const proj = arg("project") ?? process.cwd();
const dir = arg("dir") ?? ".planning";
if (!/^\.?[A-Za-z0-9_-]+$/.test(dir) || dir === "docs") {
  console.error(`private-mode: --dir must be a simple folder name other than docs (got "${dir}")`);
  process.exit(1);
}
const p = (...a: string[]): string => join(proj, ...a);
if (!existsSync(p(".claude", "kit.json"))) {
  console.error("private-mode: .claude/kit.json not found; run this in an agentic-kit project");
  process.exit(1);
}

const act = (msg: string, fn: () => void): void => {
  console.log(`${dry ? "[dry-run] " : ""}${msg}`);
  if (!dry) fn();
};
const walk = (d: string, ext: string, out: string[] = []): string[] => {
  if (!existsSync(d)) return out;
  for (const e of readdirSync(d, { withFileTypes: true })) {
    if (["node_modules", ".git", "worktrees"].includes(e.name)) continue;
    const f = join(d, e.name);
    if (e.isDirectory()) walk(f, ext, out);
    else if (e.name.endsWith(ext)) out.push(f);
  }
  return out;
};

// 0. backup
const bk = ["docs", ".claude", "CLAUDE.md", "scripts/agentic"].filter((x) => existsSync(p(x)));
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
act(`backup ${bk.join(", ")} -> logs/private-mode-backup-${stamp}.tgz`, () => {
  mkdirSync(p("logs"), { recursive: true });
  execFileSync("tar", ["-czf", p("logs", `private-mode-backup-${stamp}.tgz`), "-C", proj, ...bk]);
});

// 1. move planning files docs/ -> <dir>/
act(`create ${dir}/`, () => mkdirSync(p(dir), { recursive: true }));
for (const n of PLANNING) {
  if (!existsSync(p("docs", n))) continue;
  if (existsSync(p(dir, n))) {
    console.log(`  skip ${n}: already in ${dir}/`);
    continue;
  }
  act(`move docs/${n} -> ${dir}/${n}`, () => renameSync(p("docs", n), p(dir, n)));
}

// 2. rewrite references to the moved files (public docs keep their own paths)
const names = PLANNING.map((n) => n.replace(/\./g, "\\.")).join("|");
const re = new RegExp(`(?<![\\w./-])docs/(${names})(?![\\w-])`, "g");
const targets = [
  ...walk(p(".claude", "agents"), ".md"),
  ...walk(p(".claude", "commands"), ".md"),
  ...walk(p(dir), ".md"),
  ...walk(p("docs"), ".md"),
  ...(existsSync(p("CLAUDE.md")) ? [p("CLAUDE.md")] : []),
];
let changed = 0;
for (const f of targets) {
  const t = readFileSync(f, "utf8");
  const n = t.replace(re, `${dir}/$1`);
  if (n !== t) {
    changed++;
    if (!dry) writeFileSync(f, n);
  }
}
console.log(`${dry ? "[dry-run] " : ""}rewrote planning paths in ${changed} file(s)`);

// 3. kit.json, hooks, commands
const cfgPath = p(".claude", "kit.json");
const cfg = JSON.parse(readFileSync(cfgPath, "utf8")) as Record<string, unknown>;
cfg.private_dir = dir;
cfg.private_paths = PRIVATE_PATHS(dir);
cfg.protected_paths = ((cfg.protected_paths as string[] | undefined) ?? []).map((x) =>
  x === "docs/release/APPROVED" ? `${dir}/release/APPROVED` : x,
);
act(`kit.json: private_dir=${dir}, private_paths, protected release path`, () =>
  writeFileSync(cfgPath, `${JSON.stringify(cfg, null, 2)}\n`),
);

const patchFile = (rel: string, from: string | RegExp, to: string, already: string): void => {
  const f = p(rel);
  if (!existsSync(f)) return;
  const t = readFileSync(f, "utf8");
  if (t.includes(already) || !(typeof from === "string" ? t.includes(from) : from.test(t))) return;
  act(`patch ${rel}`, () => writeFileSync(f, t.replace(from, to)));
};
const esc = dir.replace(/\./g, "\\.");
patchFile(".claude/hooks/post-edit.sh", "| docs/*) exit 0", `| docs/* | ${dir}/*) exit 0`, `${dir}/*`);
patchFile(".claude/hooks/verify.sh", "(docs|logs)/", `(docs|logs|${esc})/`, `(docs|logs|${esc})/`);
patchFile(
  ".claude/commands/autopilot.md",
  "(worktrees/ is gitignored).",
  "(worktrees/ is gitignored). Right after `git worktree add`, run `node scripts/agentic/link-private.ts worktrees/t-<id>-<slug>`; if it exits non-zero, stop: private files must never show in git status.",
  "link-private.ts",
);
patchFile(
  ".claude/commands/build.md",
  "follow steps 2-4 of /autopilot for this single ticket",
  "follow steps 2-4 of /autopilot for this single ticket (if you use a worktree, run `node scripts/agentic/link-private.ts <worktree>` right after creating it)",
  "link-private.ts",
);

// 4. hide from git locally (shared by all worktrees); nothing private is named in the repo
const common = execFileSync("git", ["-C", proj, "rev-parse", "--git-common-dir"], { encoding: "utf8" }).trim();
const excl = join(common.startsWith("/") ? common : join(proj, common), "info", "exclude");
const wanted = [...PRIVATE_PATHS(dir), "agentic-kit", "agentic_kit"].map((x) => `/${x}`);
const have = existsSync(excl) ? readFileSync(excl, "utf8").split("\n") : [];
const add = wanted.filter((x) => !have.includes(x));
if (add.length > 0) {
  act(`.git/info/exclude += ${add.join(" ")}`, () => {
    mkdirSync(dirname(excl), { recursive: true });
    appendFileSync(excl, `${have.length > 0 && have.at(-1) !== "" ? "\n" : ""}${add.join("\n")}\n`);
  });
}

// 5. verify nothing private is visible to git
if (!dry && existsSync(p("scripts", "agentic", "link-private.ts"))) {
  execFileSync(process.execPath, [p("scripts", "agentic", "link-private.ts"), "--check", proj], { stdio: "inherit" });
}
if (existsSync(p(".gitignore")) && /^docs\/?$/m.test(readFileSync(p(".gitignore"), "utf8"))) {
  console.log("note: .gitignore still ignores docs/. Remove that line so the public docs can be tracked.");
}
console.log(`${dry ? "[dry-run] done, nothing changed" : "private mode on"}: planning in ${dir}/, public docs in docs/`);
if (statSync(proj).isDirectory() && !dry) console.log(`backup: logs/private-mode-backup-${stamp}.tgz`);
