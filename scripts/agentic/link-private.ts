// Private mode: make the local-only files (planning dir, .claude, CLAUDE.md, logs) available inside a git
// worktree and prove none of them can reach a commit.
//   node scripts/agentic/link-private.ts <worktree-dir>    symlink the private paths in, then verify
//   node scripts/agentic/link-private.ts --check [<dir>]   verify only (also run it in the main checkout)
// Exit 0 = ok or private mode not enabled. Exit 1 = a private path is tracked, staged or shows in git status.
// Why symlinks plus .git/info/exclude: a worktree only has tracked files, and a `dir/` pattern in .gitignore
// does not match a symlink, so slashless patterns go in the shared info/exclude (read by every worktree).
import { execFileSync } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readlinkSync,
  symlinkSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";

const DEFAULT_PRIVATE = [".planning", ".claude", "CLAUDE.md", "logs"];

const git = (cwd: string, ...a: string[]): string =>
  execFileSync("git", ["-C", cwd, ...a], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });

const args = process.argv.slice(2);
const check = args.includes("--check");
const target = resolve(args.find((a) => !a.startsWith("--")) ?? process.cwd());

const common = resolve(target, git(target, "rev-parse", "--git-common-dir").trim());
const mainRoot = dirname(common);

interface Cfg {
  private_dir?: string;
  private_paths?: string[];
}
let cfg: Cfg = {};
try {
  cfg = JSON.parse(readFileSync(join(mainRoot, ".claude", "kit.json"), "utf8")) as Cfg;
} catch {
  /* no kit.json: private mode is off */
}
if (!cfg.private_dir) {
  console.log("link-private: private mode is not enabled (no private_dir in .claude/kit.json); nothing to do");
  process.exit(0);
}
const paths = cfg.private_paths ?? DEFAULT_PRIVATE;
const isPrivate = (f: string): boolean => paths.some((p) => f === p || f.startsWith(`${p}/`));

// 1. slashless patterns in the shared exclude file (idempotent)
const excludeFile = join(common, "info", "exclude");
mkdirSync(dirname(excludeFile), { recursive: true });
const have = existsSync(excludeFile) ? readFileSync(excludeFile, "utf8").split("\n") : [];
const missing = paths.map((p) => `/${p}`).filter((l) => !have.includes(l));
if (missing.length > 0) appendFileSync(excludeFile, `${have.at(-1) === "" || have.length === 0 ? "" : "\n"}${missing.join("\n")}\n`);

// 2. link (worktrees only)
if (!check && target !== mainRoot) {
  for (const p of paths) {
    const src = join(mainRoot, p);
    const dst = join(target, p);
    if (!existsSync(src)) continue;
    let st;
    try {
      st = lstatSync(dst);
    } catch {
      st = undefined;
    }
    if (st) {
      if (st.isSymbolicLink() && resolve(dirname(dst), readlinkSync(dst)) === src) continue;
      console.error(`link-private: ${dst} already exists and is not the expected link; leaving it alone`);
      continue;
    }
    symlinkSync(src, dst);
    console.log(`linked ${p}`);
  }
}

// 3. verify: nothing private is tracked, staged, or visible to git status
const tracked = git(target, "ls-files").split("\n").filter(Boolean).filter(isPrivate);
const visible = git(target, "status", "--porcelain", "--untracked-files=normal")
  .split("\n")
  .filter(Boolean)
  .map((l) => l.slice(3).replace(/^"|"$/g, ""))
  .filter(isPrivate);
const bad = [...new Set([...tracked, ...visible])];
if (bad.length > 0) {
  console.error(`link-private: PRIVATE FILES WOULD BE EXPOSED in ${target}:\n  ${bad.join("\n  ")}`);
  console.error("Do not commit. Unstage with `git rm --cached -r <path>` and check .git/info/exclude.");
  process.exit(1);
}
console.log(`link-private: ok (${paths.join(", ")} stay out of git) in ${target}`);
