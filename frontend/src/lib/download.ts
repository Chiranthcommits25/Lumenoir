/** Download helpers. Results are delivered in-memory; nothing is stored. */

function triggerDownload(href: string, filename: string): void {
  const link = document.createElement('a')
  link.href = href
  link.download = filename
  link.rel = 'noopener'
  document.body.append(link)
  link.click()
  link.remove()
}

export function downloadObjectUrl(url: string, filename: string): void {
  triggerDownload(url, filename)
}

export function downloadBase64(
  data: string,
  mediaType: string,
  filename: string,
): void {
  triggerDownload(`data:${mediaType};base64,${data}`, filename)
}

/** ``photo.jpeg`` -> ``lumenoir-photo.png`` style names for downloads. */
export function downloadName(filename: string, suffix = '', extension = 'png'): string {
  const stem = filename.replace(/\.[^.]+$/, '') || 'image'
  const tail = suffix ? `-${suffix}` : ''
  return `lumenoir-${stem}${tail}.${extension}`
}
