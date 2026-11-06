/** Formatting helpers for the small number of values the UI displays. */

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${bytes} B`

  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unitIndex = 0

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex += 1
  }

  const decimals = value >= 10 ? 0 : 1
  return `${value.toFixed(decimals)} ${units[unitIndex]}`
}

export function formatExtensions(extensions: string[]): string {
  return extensions
    .map((extension) => extension.replace('.', '').toUpperCase())
    .join(', ')
}

export function formatMegapixels(megapixels: number): string {
  return `${megapixels.toFixed(2)} MP`
}
