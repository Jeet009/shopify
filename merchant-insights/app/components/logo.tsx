/**
 * Merchant Insights brand mark. A rounded-square app icon with a bar-chart +
 * insight spark. Scales cleanly; uses a fixed brand gradient (independent of
 * Polaris theme) so it reads as a logo, not UI chrome.
 */
export function Logo({ size = 40 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Merchant Insights"
    >
      <defs>
        <linearGradient id="mi-grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop stopColor="#6D5BFF" />
          <stop offset="1" stopColor="#9C3FE4" />
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="12" fill="url(#mi-grad)" />
      {/* bars */}
      <rect x="12" y="26" width="6" height="10" rx="2" fill="#FFFFFF" opacity="0.92" />
      <rect x="21" y="21" width="6" height="15" rx="2" fill="#FFFFFF" opacity="0.92" />
      <rect x="30" y="16" width="6" height="20" rx="2" fill="#FFFFFF" opacity="0.55" />
      {/* insight spark */}
      <circle cx="33" cy="13" r="3.4" fill="#FFE27A" />
    </svg>
  );
}
