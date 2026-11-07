import type { PipelineArtifact } from '../lib/api'

export type IntermediateGalleryProps = {
  artifacts: PipelineArtifact[]
  className?: string
}

type ArtifactState = 'available' | 'not_applicable' | 'pending'

function artifactState(artifact: PipelineArtifact): ArtifactState {
  if (artifact.available && artifact.data && artifact.media_type) return 'available'
  if (artifact.applicable === false) return 'not_applicable'
  return 'pending'
}

const STATE_LABEL: Record<ArtifactState, string> = {
  available: 'Available',
  not_applicable: 'Not applicable',
  pending: 'Pending',
}

const STATE_CLASS: Record<ArtifactState, string> = {
  available: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300',
  not_applicable: 'border-ink-600 bg-ink-800/60 text-mist-400',
  pending: 'border-accent/30 bg-accent/10 text-accent-soft',
}

/**
 * Explainable-AI panel. Renders real pixels only when the backend supplies
 * them. Mode-aware: artifacts not part of the selected mode are marked
 * "Not applicable"; applicable artifacts without data are "Pending".
 * No placeholder imagery is generated.
 */
export function IntermediateGallery({
  artifacts,
  className = '',
}: IntermediateGalleryProps) {
  if (!artifacts.length) {
    return (
      <p className="text-[15px] leading-relaxed text-mist-500 sm:text-base">
        Intermediate outputs are not available yet.
      </p>
    )
  }

  return (
    <ul className={`grid gap-3 sm:grid-cols-2 xl:grid-cols-3 ${className}`}>
      {artifacts.map((artifact) => {
        const state = artifactState(artifact)
        const src =
          state === 'available' && artifact.data && artifact.media_type
            ? `data:${artifact.media_type};base64,${artifact.data}`
            : null

        return (
          <li
            key={artifact.kind}
            className={`overflow-hidden rounded-xl border bg-ink-900 ${
              state === 'not_applicable' ? 'border-ink-700/70' : 'border-ink-700'
            }`}
          >
            <div className="grid grid-cols-1 gap-x-2 gap-y-1.5 border-b border-ink-700 px-3.5 py-2.5 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start">
              <p className="min-w-0 text-sm font-medium leading-snug text-mist-200">
                {artifact.label}
              </p>
              <span
                className={`inline-flex w-fit max-w-full items-center justify-self-start whitespace-normal break-words rounded-full border px-2.5 py-1 text-[11px] leading-tight tracking-[0.1em] uppercase sm:justify-self-end ${STATE_CLASS[state]}`}
              >
                {STATE_LABEL[state]}
              </span>
            </div>

            {src ? (
              <img
                src={src}
                alt={`${artifact.label} output`}
                className="max-h-48 w-full bg-ink-950 object-contain"
              />
            ) : (
              <div className="flex min-h-24 flex-col justify-center gap-1 px-3.5 py-4">
                <p
                  className={`text-sm leading-relaxed ${
                    state === 'not_applicable' ? 'text-mist-500' : 'text-mist-400'
                  }`}
                >
                  {state === 'not_applicable'
                    ? 'Not part of the selected enhancement mode.'
                    : (artifact.reason ?? artifact.description)}
                </p>
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}
