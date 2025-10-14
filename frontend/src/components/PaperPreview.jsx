export default function PaperPreview({ url }) {
  if (!url) return null;

  return (
    <div className="glass rounded-2xl p-6 md:p-8 mt-6">
      <h3 className="text-lg font-semibold mb-3">Your Paper is Ready</h3>

      {/* === Download Link === */}
      <div className="flex items-center gap-3">
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="px-4 py-2 rounded-xl bg-emerald-500/20 text-emerald-200 border border-emerald-400/30 hover:bg-emerald-500/30"
        >
          ⬇️ Download DOCX
        </a>
        <span className="text-slate-400 text-sm">
          If it doesn’t open, right–click → <em>Save link as…</em>
        </span>
      </div>

      {/* === Inline Preview (best effort, may not render DOCX in all browsers) === */}
      <div className="mt-5 rounded-xl overflow-hidden border border-slate-700/40">
        <iframe
          src={`https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(
            url
          )}`}
          className="w-full h-[70vh] bg-slate-950"
          title="Paper Preview"
        ></iframe>
      </div>
    </div>
  );
}
