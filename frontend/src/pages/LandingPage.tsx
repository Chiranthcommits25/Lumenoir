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
    status: 'Implemented',
    active: true,
  },
  {
    title: 'Enhancement pipeline',
    body: 'Classical, deep learning and hybrid modes run through the backend pipeline. Mode selection controls which stages and intermediate outputs are produced.',
    status: 'Available',
    active: true,
  },
  {
    title: 'Explainable outputs',
    body: 'Illumination, noise, classical, deep learning and fusion panels are wired end to end, and only ever show images the backend really produced.',
    status: 'Mode-aware',
    active: true,
  },
]

export function LandingPage({ onOpenWorkspace }: LandingPageProps) {
  return (
    <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8">
      <section className="py-14 sm:py-20 lg:py-28">
        <div className="flex max-w-2xl flex-col items-start gap-6">
            <span className="inline-flex items-center gap-2 rounded-full border border-ink-600 px-3.5 py-1.5 text-xs tracking-[0.18em] text-mist-400 uppercase sm:text-[13px]">
              <span
                aria-hidden="true"
                className="h-1.5 w-1.5 rounded-full bg-accent"
              />
              Image restoration workspace
            </span>

          <h1 className="text-4xl leading-[1.08] font-medium tracking-tight text-mist-50 sm:text-5xl lg:text-6xl">
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
              className="inline-flex min-h-12 items-center justify-center rounded-full bg-mist-50 px-7 text-[15px] font-medium text-ink-950 transition-colors outline-none hover:bg-white focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-ink-950 sm:text-base"
            >
              Open the workspace
            </button>
            <span className="text-sm leading-relaxed text-mist-500">
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
            className="flex min-w-0 flex-col gap-3 rounded-2xl border border-ink-700 bg-ink-850 p-5"
          >
            <div className="grid grid-cols-1 gap-x-3 gap-y-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
              <h2 className="min-w-0 text-sm font-medium leading-snug text-mist-50 sm:text-base">
                {capability.title}
              </h2>
              <span
                className={`inline-flex w-fit max-w-full items-center justify-self-start whitespace-normal break-words rounded-full border px-2.5 py-1 text-[11px] leading-tight tracking-[0.12em] uppercase sm:justify-self-end ${
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
          <h2 className="text-sm font-medium text-mist-50 sm:text-base">
            How to use the workspace
          </h2>
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-mist-400">
            Upload a JPG, JPEG or PNG image, choose Classical, Deep Learning or
            Hybrid mode, then run enhancement. Compare the original and enhanced
            images with the slider, inspect available intermediate outputs and
            download the result. If the backend is unavailable or a request fails,
            the interface reports the error and keeps the last valid state.
          </p>
        </div>
      </section>
    </div>
  )
}
