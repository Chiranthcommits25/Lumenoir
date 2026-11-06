export type LandingPageProps = {
  onOpenWorkspace: () => void
}

const CAPABILITIES = [
  {
    title: 'Validated uploads',
    body: 'File type, size, dimensions and integrity are checked before anything is processed. Corrupted or spoofed files are rejected with a clear reason.',
    status: 'Implemented',
    active: true,
  },
  {
    title: 'Interactive comparison',
    body: 'A touch-friendly before/after slider for inspecting a result at any position, with keyboard support.',
    status: 'Implemented · placeholder imagery',
    active: true,
  },
  {
    title: 'Model-ready pipeline',
    body: 'Stages, a model registry, load-once weights handling and safe resizing are in place; the algorithms that fill them are still pending research.',
    status: 'Architecture ready',
    active: false,
  },
  {
    title: 'Explainable outputs',
    body: 'Illumination, noise, classical, deep learning and fusion panels are wired end to end, and only ever show images the backend really produced.',
    status: 'Wired · awaiting outputs',
    active: true,
  },
]

export function LandingPage({ onOpenWorkspace }: LandingPageProps) {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8">
      <section className="py-14 sm:py-20 lg:py-28">
        <div className="flex max-w-2xl flex-col items-start gap-6">
          <span className="inline-flex items-center gap-2 rounded-full border border-ink-600 px-3 py-1 text-[11px] tracking-[0.18em] text-mist-400 uppercase">
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 rounded-full bg-accent"
            />
            Foundation build
          </span>

          <h1 className="text-4xl leading-[1.05] font-medium tracking-tight text-mist-50 sm:text-5xl lg:text-6xl">
            Light, restored.
            <br />
            <span className="text-mist-400">Detail, preserved.</span>
          </h1>

          <p className="max-w-xl text-base leading-relaxed text-mist-400 sm:text-lg">
            LUMENOIR is an illumination-adaptive workspace for underexposed,
            noisy and colour-distorted photographs — designed to stay
            lightweight enough for affordable, CPU-based deployment.
          </p>

          <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row sm:items-center">
            <button
              type="button"
              onClick={onOpenWorkspace}
              className="inline-flex min-h-11 items-center justify-center rounded-full bg-mist-50 px-6 text-sm font-medium text-ink-950 transition-colors outline-none hover:bg-white focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-ink-950"
            >
              Open the workspace
            </button>
            <span className="text-xs text-mist-500 sm:text-sm">
              JPG, JPEG or PNG · up to 10 MB
            </span>
          </div>
        </div>
      </section>

      <section
        aria-label="Current capabilities"
        className="grid gap-3 border-t border-ink-700/70 py-10 sm:grid-cols-2 lg:grid-cols-3"
      >
        {CAPABILITIES.map((capability) => (
          <article
            key={capability.title}
            className="flex flex-col gap-3 rounded-2xl border border-ink-700 bg-ink-850 p-5"
          >
            <div className="flex items-start justify-between gap-3">
              <h2 className="text-sm font-medium text-mist-50">
                {capability.title}
              </h2>
              <span
                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] tracking-[0.12em] uppercase ${
                  capability.active
                    ? 'border-accent/30 bg-accent/10 text-accent-soft'
                    : 'border-ink-600 text-mist-500'
                }`}
              >
                {capability.status}
              </span>
            </div>
            <p className="text-sm leading-relaxed text-mist-400">
              {capability.body}
            </p>
          </article>
        ))}
      </section>

      <section className="border-t border-ink-700/70 py-10">
        <div className="rounded-2xl border border-ink-700 bg-ink-900 p-5 sm:p-6">
          <h2 className="text-sm font-medium text-mist-50">
            What this build does not do yet
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-mist-400">
            No enhancement has been implemented or simulated. The Retinex
            variant, denoising method, deep learning architecture, dataset and
            training strategy, adaptive fusion weights, parameter selection and
            evaluation methodology are still open research decisions. Those
            modules exist as documented placeholders only.
          </p>
        </div>
      </section>
    </div>
  )
}
