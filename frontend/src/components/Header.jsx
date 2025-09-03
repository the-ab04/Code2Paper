export default function Header(){
  return (
    <header className="px-6 py-5 sticky top-0 z-20 bg-gradient-to-b from-slate-900/70 to-transparent backdrop-blur-xs">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-2xl bg-indigo-500/20 border border-indigo-300/20 grid place-items-center">
            <span className="text-indigo-300 font-black">C2P</span>
          </div>
          <h1 className="text-xl md:text-2xl font-semibold tracking-tight">
            Code<span className="text-indigo-300">2</span>Paper
          </h1>
        </div>
        <nav className="text-sm text-slate-300">
          <a href="https://example.com" className="hover:text-white transition-colors">Docs</a>
        </nav>
      </div>
    </header>
  );
}
