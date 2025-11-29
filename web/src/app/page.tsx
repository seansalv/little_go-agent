import Link from "next/link";

const projects = [
  {
    title: "Play vs Agent",
    summary: "Live 5×5 Go demo powered by the alpha-beta bot. Enforces KO, logs every turn, and pings the FastAPI backend.",
    href: "/projects/go-agent",
    cta: "Play vs Agent",
    badge: "New",
  },
  {
    title: "Post-mortem + heuristics",
    summary: "Notebook covering evaluation weights, killer moves, and rollout pruning decisions that shaped the bot.",
    href: "https://github.com/seansalv",
    cta: "Read notes",
    badge: "Deep dive",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-5xl px-6 py-16 sm:px-10">
        <header className="space-y-6">
          <p className="text-xs uppercase tracking-[0.3em] text-sky-400">Little Go demo</p>
          <h1 className="text-4xl font-semibold leading-tight text-white sm:text-5xl">
            Alpha-beta experiments for a tiny Go board
          </h1>
          <p className="max-w-3xl text-lg text-slate-300">
            A dark-mode playground for the bot I’ve been tuning for class.
            The hero card below jumps straight into the interactive match against Render-hosted FastAPI,
            while the rest collect write-ups and build notes.
          </p>
          <div className="flex flex-wrap gap-4">
            <Link
              href="/projects/go-agent"
              className="inline-flex items-center gap-2 rounded-full bg-sky-400 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-sky-300"
            >
              Play vs Agent →
            </Link>
            <a
              href="https://github.com/seansalv"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-slate-700 px-6 py-3 text-sm font-semibold text-slate-100 transition hover:border-sky-200 hover:text-sky-200"
            >
              GitHub profile
            </a>
          </div>
        </header>

        <section className="mt-14 grid gap-6 md:grid-cols-2">
          {projects.map((project) => (
            <div
              key={project.title}
              className="flex h-full flex-col gap-4 rounded-3xl border border-slate-800/70 bg-slate-900/40 p-6"
            >
              <div className="flex items-center justify-between text-sm text-slate-400">
                <span>{project.badge}</span>
                <span>5×5 Go</span>
              </div>
              <h2 className="text-2xl font-semibold text-white">{project.title}</h2>
              <p className="text-sm text-slate-300">{project.summary}</p>
              <div className="mt-auto">
                {project.href.startsWith("/") ? (
                  <Link
                    href={project.href}
                    className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-sky-200"
                  >
                    {project.cta} →
                  </Link>
                ) : (
                  <a
                    href={project.href}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-2 rounded-full border border-slate-600 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-sky-200 hover:text-sky-200"
                  >
                    {project.cta} ↗
                  </a>
                )}
              </div>
            </div>
          ))}
        </section>
      </div>
    </main>
  );
}
