import React, { useState, useRef, useCallback } from 'react';
import { Upload, FileUp, Zap, ScanEye, Code, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import JSONPretty from 'react-json-pretty';
import 'react-json-pretty/themes/monikai.css';

const API_BASE = '/api/ocr';

function LegalOCR() {
    const [file, setFile] = useState(null);
    const [previewData, setPreviewData] = useState(null);
    const [isPreviewLoading, setIsPreviewLoading] = useState(false);

    const [mode, setMode] = useState('Fast'); // 'Fast' or 'Accurate'
    const [promptJson, setPromptJson] = useState('{\n  "so_hieu": "",\n  "ngay": "",\n  "noi_dung": ""\n}');

    const [isExtracting, setIsExtracting] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const fileInputRef = useRef(null);

    const handleFileSelect = async (e) => {
        const selected = e.target.files[0];
        if (selected) {
            setFile(selected);
            setResults(null);
            setError(null);
            await generatePreview(selected);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleDrop = async (e) => {
        e.preventDefault();
        const selected = e.dataTransfer.files[0];
        if (selected && (selected.type === 'application/pdf' || selected.type.startsWith('image/'))) {
            setFile(selected);
            setResults(null);
            setError(null);
            await generatePreview(selected);
        }
    };

    const generatePreview = async (selectedFile) => {
        setIsPreviewLoading(true);
        setPreviewData(null);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const res = await fetch(`${API_BASE}/preview`, {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();
            if (data.success) {
                setPreviewData(`data:image/png;base64,${data.image_b64}`);
            } else {
                setError(data.error || 'Failed to generate preview');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsPreviewLoading(false);
        }
    };

    const handleExtract = async () => {
        if (!file) return;

        setIsExtracting(true);
        setResults(null);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('mode', mode);
            formData.append('prompt_json', promptJson);

            const res = await fetch(`${API_BASE}/extract`, {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();
            if (data.success) {
                setResults(data.results);
            } else {
                setError(data.error || 'Extraction failed');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setIsExtracting(false);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full bg-slate-50 dark:bg-slate-900 overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-primary-200 dark:border-primary-700 bg-white dark:bg-slate-800">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center">
                        <ScanEye className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold text-slate-800 dark:text-white">OCR Debug Preview</h1>
                        <p className="text-sm text-slate-500 dark:text-slate-400">TrÃ­ch xuáº¥t thá»±c thá»ƒ tá»« tÃ i liá»‡u phÃ¡p lÃ½ sang JSON</p>
                    </div>
                </div>

                {file && (
                    <button
                        onClick={handleExtract}
                        disabled={isExtracting}
                        className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded-lg font-medium transition-colors"
                    >
                        {isExtracting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Code className="w-5 h-5" />}
                        {isExtracting ? 'Äang trÃ­ch xuáº¥t...' : 'Cháº¡y TrÃ­ch Xuáº¥t'}
                    </button>
                )}
            </div>

            <div className="flex-1 flex overflow-hidden">
                {/* Left column: Upload & Preview */}
                <div className="w-1/2 p-6 flex flex-col border-r border-slate-200 dark:border-slate-700 overflow-y-auto">
                    {!file ? (
                        <div
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                            className="flex-1 border-2 border-dashed border-slate-300 dark:border-slate-600 hover:border-indigo-500 dark:hover:border-indigo-400 rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-colors bg-white dark:bg-slate-800"
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,image/png,image/jpeg"
                                onChange={handleFileSelect}
                                className="hidden"
                            />
                            <FileUp className="w-16 h-16 text-slate-400 mb-4" />
                            <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-200">KÃ©o tháº£ tÃ i liá»‡u vÃ o Ä‘Ã¢y</h3>
                            <p className="text-slate-500 dark:text-slate-400 mt-2">Há»— trá»£ PDF, PNG, JPG (Windows Optimized)</p>
                        </div>
                    ) : (
                        <div className="flex-1 flex flex-col">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="font-semibold text-slate-700 dark:text-slate-200">Báº£n xem trÆ°á»›c ({file.name})</h3>
                                <button onClick={() => { setFile(null); setPreviewData(null); setResults(null); }} className="text-sm text-red-500 hover:text-red-600">
                                    Táº£i file khÃ¡c
                                </button>
                            </div>

                            <div className="flex-1 bg-slate-200 dark:bg-slate-950 rounded-xl overflow-hidden flex items-center justify-center border border-slate-300 dark:border-slate-700 relative">
                                {isPreviewLoading ? (
                                    <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
                                ) : previewData ? (
                                    <img src={previewData} alt="Preview" className="max-w-full max-h-full object-contain" />
                                ) : (
                                    <span className="text-slate-500">Preview not available</span>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Right column: Settings & Results */}
                <div className="w-1/2 p-6 flex flex-col overflow-y-auto bg-white dark:bg-slate-800">
                    {/* Settings */}
                    <div className="mb-6">
                        <h3 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">Cháº¿ Ä‘á»™ xá»­ lÃ½</h3>
                        <div className="flex gap-4 mb-4">
                            <button
                                onClick={() => setMode('Fast')}
                                className={clsx(
                                    "flex-1 flex items-center justify-center gap-2 py-3 rounded-lg border-2 transition-all font-medium",
                                    mode === 'Fast'
                                        ? "border-indigo-500 bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300"
                                        : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-400 hover:border-slate-300"
                                )}
                            >
                                <Zap className="w-5 h-5" /> Fast (Text Direct)
                            </button>
                            <button
                                onClick={() => setMode('Accurate')}
                                className={clsx(
                                    "flex-1 flex items-center justify-center gap-2 py-3 rounded-lg border-2 transition-all font-medium",
                                    mode === 'Accurate'
                                        ? "border-purple-500 bg-purple-50 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300"
                                        : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-400 hover:border-slate-300"
                                )}
                            >
                                <ScanEye className="w-5 h-5" /> Accurate (Vision OCR)
                            </button>
                        </div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
                            {mode === 'Fast'
                                ? "DÃ¹ng cho PDF sá»‘ (cÃ³ sáºµn text layer). Nhanh, nháº¹, tiáº¿t kiá»‡m tÃ i nguyÃªn."
                                : "DÃ¹ng cho PDF scan, áº£nh chá»¥p, cÃ³ dáº¥u má»™c. Gá»­i trá»±c tiáº¿p áº£nh cho Vision Model xá»­ lÃ½."}
                        </p>

                        <h3 className="font-semibold text-slate-700 dark:text-slate-200 mb-2">Cáº¥u trÃºc JSON mong muá»‘n</h3>
                        <textarea
                            value={promptJson}
                            onChange={(e) => setPromptJson(e.target.value)}
                            className="w-full h-32 px-3 py-2 bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-sm text-slate-800 dark:text-slate-200 font-mono focus:ring-2 focus:ring-indigo-500 outline-none resize-none"
                            spellCheck="false"
                        />
                    </div>

                    {/* Results */}
                    <div className="flex-1 flex flex-col">
                        <h3 className="font-semibold text-slate-700 dark:text-slate-200 mb-3">Káº¿t quáº£ JSON</h3>

                        {error && (
                            <div className="p-4 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg text-sm mb-4">
                                {error}
                            </div>
                        )}

                        <div className="flex-1 bg-[#272822] rounded-xl border border-slate-300 dark:border-slate-700 overflow-auto">
                            {results ? (
                                <div className="p-4">
                                    <JSONPretty id="json-pretty" data={results} />
                                </div>
                            ) : (
                                <div className="h-full flex items-center justify-center text-slate-500 italic pb-8">
                                    {isExtracting ? "Äang xá»­ lÃ½ trÃªn RTX 4090..." : "ChÆ°a cÃ³ dá»¯ liá»‡u"}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LegalOCR;

