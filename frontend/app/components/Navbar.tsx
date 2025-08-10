export function Navbar() {
  return (
    <div className="w-full flex items-center justify-between py-4 px-6 border-b border-gray-800/60 bg-[#0b1220]">
      <div className="flex items-center gap-2">
        <div className="w-2.5 h-2.5 rounded-sm bg-cyan-400" />
        <span className="text-sm text-gray-300 tracking-wide">CallGuard</span>
      </div>
      <div className="flex items-center gap-6 text-sm text-gray-400">
        <span className="cursor-default">Dashboard</span>
        <span className="cursor-default">History</span>
        <span className="cursor-default">Settings</span>
      </div>
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-full bg-gray-700/60" />
        <div className="w-9 h-9 rounded-full bg-gray-700/60" />
      </div>
    </div>
  );
}


