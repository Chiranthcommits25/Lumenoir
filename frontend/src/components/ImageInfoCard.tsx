import type { ImageInfo } from '../lib/api'
import { formatBytes, formatMegapixels } from '../lib/format'

export type ImageInfoCardProps = {
  info: ImageInfo
}

/** Basic information and measurements reported for a validated upload. */
export function ImageInfoCard({ info }: ImageInfoCardProps) {
  const rows: { label: string; value: string }[] = [
    { label: 'Format', value: info.format },
    { label: 'Dimensions', value: `${info.width} × ${info.height} px` },
    { label: 'File size', value: formatBytes(info.size_bytes) },
    { label: 'Megapixels', value: formatMegapixels(info.megapixels) },
    { label: 'Colour mode', value: info.colour_mode },
    { label: 'Aspect ratio', value: `${info.aspect_ratio.toFixed(2)} : 1` },
  ]

  const statistics = info.statistics
  if (statistics) {
    rows.push(
      { label: 'Mean luminance', value: statistics.luminance.mean.toFixed(1) },
      { label: 'Luminance spread', value: `±${statistics.luminance.stddev.toFixed(1)}` },
      {
        label: 'Luminance range',
        value: `${statistics.luminance.min} – ${statistics.luminance.max}`,
      },
    )
  }

  const histogram = statistics?.luminance_histogram ?? []
  const peak = histogram.length ? Math.max(...histogram) : 0

  return (
    <section className="rounded-2xl border border-ink-700 bg-ink-850 p-4 sm:p-5">
      <div className="grid grid-cols-1 gap-x-3 gap-y-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
        <div className="min-w-0">
          <h2 className="text-sm font-medium leading-snug text-mist-50 sm:text-base">Image information</h2>
          <p className="mt-1 truncate text-sm text-mist-500" title={info.filename}>
            {info.filename}
          </p>
        </div>
        <span className="inline-flex w-fit max-w-full items-center justify-self-start whitespace-normal break-words rounded-full border border-accent/30 bg-accent/10 px-2.5 py-1 text-[11px] leading-tight tracking-[0.14em] text-accent-soft uppercase sm:justify-self-end">
          Valid
        </span>
      </div>

      <dl className="mt-4 divide-y divide-ink-700/80 text-sm">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-baseline justify-between gap-4 py-2"
          >
            <dt className="text-mist-500">{row.label}</dt>
            <dd className="text-right font-medium text-mist-200">{row.value}</dd>
          </div>
        ))}
      </dl>

      {histogram.length ? (
        <figure className="mt-4">
          <figcaption className="text-sm text-mist-500">
            Luminance distribution ({histogram.length} bins)
          </figcaption>
          <div
            aria-hidden="true"
            className="mt-2 flex h-12 items-end gap-0.5"
            title="Measured luminance histogram"
          >
            {histogram.map((count, index) => (
              <div
                key={index}
                style={{ height: `${peak ? Math.max(2, (count / peak) * 100) : 2}%` }}
                className="flex-1 rounded-sm bg-accent/45"
              />
            ))}
          </div>
        </figure>
      ) : null}

      <p className="mt-3 text-sm leading-relaxed text-mist-500">
        Nothing is stored: the image is validated in memory and discarded.
      </p>
    </section>
  )
}
