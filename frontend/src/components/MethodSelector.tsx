import type { EnhancementMode, ModeCapability } from '../lib/api'

export type MethodSelectorProps = {
  value: EnhancementMode
  modes: ModeCapability[]
  onChange: (mode: EnhancementMode) => void
  disabled?: boolean
}

/**
 * Segmented method picker. Unavailable modes stay selectable on purpose: the
 * API then answers with the exact pending stages, which is more informative
 * than a dead control.
 */
export function MethodSelector({
  value,
  modes,
  onChange,
  disabled = false,
}: MethodSelectorProps) {
  const options = modes.length
    ? modes
    : [
        { mode: 'classical' as const, label: 'Classical CV', available: false, required_stages: [] },
        { mode: 'deep_learning' as const, label: 'Deep learning', available: false, required_stages: [] },
        { mode: 'hybrid' as const, label: 'Hybrid', available: false, required_stages: [] },
      ]

  return (
    <div>
      <div
        role="radiogroup"
        aria-label="Enhancement method"
        className="grid grid-cols-3 gap-1 rounded-full border border-ink-700 bg-ink-900 p-1"
      >
        {options.map((option) => {
          const selected = option.mode === value
          return (
            <button
              key={option.mode}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onChange(option.mode)}
              className={`flex min-h-10 flex-col items-center justify-center rounded-full px-2 py-1.5 text-xs transition-colors outline-none focus-visible:ring-2 focus-visible:ring-accent/60 sm:text-sm ${
                selected
                  ? 'bg-ink-700 text-mist-50'
                  : 'text-mist-400 hover:text-mist-200'
              } ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
            >
              <span className="font-medium">{option.label}</span>
              <span
                className={`text-[10px] tracking-[0.1em] uppercase ${
                  option.available ? 'text-emerald-400/90' : 'text-mist-500'
                }`}
              >
                {option.available ? 'Ready' : 'Pending'}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
