export function StatusChip({ label }: { label: string }) {
  const color =
    label === "SCAM" ? "bg-red-600" : label === "SUSPICIOUS" ? "bg-yellow-500" : "bg-emerald-600";
  const text = label === "SCAM" ? "High Risk" : label === "SUSPICIOUS" ? "Caution" : "Safe";
  return (
    <div className={`w-full rounded-lg p-6 ${color} text-white`}> 
      <div className="text-xs opacity-80">{label}</div>
      <div className="text-2xl font-semibold mt-1">{text}</div>
    </div>
  );
}


