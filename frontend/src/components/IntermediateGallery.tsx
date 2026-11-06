import type { PipelineArtifact } from '../lib/api'

export type IntermediateGalleryProps = {
  artifacts: PipelineArtifact[]
  className?: string
}

/**
 * Explainable-AI panel. Renders real pixels only when the backend supplies
 * them; otherwise it states why that output does not exist yet. No placeholder
 * imagery is used here, so the panel can never be mistaken for a result.
 */
export function IntermediateGallery({
  artifacts,
  className = '',
}: IntermediateGalleryProps) {
  if (!artifacts.length) {
    return (
      <p className="text-sm text-mist-500">
        Intermediate outputs are not available yet.
      </p>
    )
  }

  return (
    <ul
      className={`grid gap-3 sm:grid-cols-2 xl:grid-cols-3 ${className}`}
    >
      {artifacts.map((artifact) => {
        const src =
          artifact.available && artifact.data && artifact.media_type
            ? `data:${artifact.media_type};base64,${artifact.data}`
            : null

        return (
          <li
            key={artifact.kind}
            className="overflow-hidden rounded-xl border border-ink-700 bg-ink-900"
          >
            <div className="flex items-center justify-between gap-2 border-b border-ink-700 px-3 py-2">
              <p className="min-w-0 truncate text-xs font-medium text-mist-200">
                {artifact.label}
              </p>
              <span
                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] tracking-[0.1em] uppercase ${
                  artifact.available
                    ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
                    : 'border-ink-600 text-mist-500'
                }`}
              >
                {artifact.available ? 'Available' : 'Pending'}
              </span>
            </div>

            {src ? (
              <img
                src={src}
                alt={`${artifact.label} output`}
                className="max-h-48 w-full bg-ink-950 object-contain"
              />
            ) : (
              <div className="flex min-h-24 flex-col justify-center gap-1 px-3 py-4">
                <p className="text-xs leading-relaxed text-mist-500">
                  {artifact.reason ?? artifact.description}
                </p>
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}
