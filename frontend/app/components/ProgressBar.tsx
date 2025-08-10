export function ProgressBar({
  value,
  color = "bg-blue-500",
  heightClass = "h-2.5",
}: {
  value: number; // 0..100
  color?: string;
  heightClass?: string;
}) {
  const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
  return (
    <div className={`w-full ${heightClass} bg-gray-700/60 rounded-full overflow-hidden`}> 
      <div
        className={`${color} ${heightClass}`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}


