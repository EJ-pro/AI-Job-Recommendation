import React, { useEffect, useState } from 'react';
import { Bike, Flag, Trophy, Sparkles } from 'lucide-react';
import ProgressBar from '@/components/ProgressBar'; // Use local component
import clsx from 'clsx';

interface ProgressHeroProps {
    progress: number; // 0 to 100
}

export function ProgressHero({ progress }: ProgressHeroProps) {
    const [animatedProgress, setAnimatedProgress] = useState(0);

    useEffect(() => {
        // Smooth animation for progress change
        const timer = setTimeout(() => setAnimatedProgress(progress), 100);
        return () => clearTimeout(timer);
    }, [progress]);

    return (
        <div className="relative w-full py-8 px-4 bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl overflow-hidden shadow-xl mb-8 border border-slate-700">
            {/* Background Decorations */}
            <div className="absolute top-0 left-0 w-full h-full opacity-20 pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-primary blur-3xl rounded-full" />
                <div className="absolute bottom-1/4 right-1/4 w-40 h-40 bg-blue-500 blur-3xl rounded-full" />
            </div>

            <div className="relative z-10 flex flex-col items-center">
                <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
                    <Trophy className="w-6 h-6 text-yellow-400" />
                    Job Readiness Tracker
                </h2>
                <p className="text-slate-400 mb-8 text-sm">
                    퀴즈를 풀고 로드맵을 완주하여 준비된 인재가 되어보세요!
                </p>

                {/* Track Container */}
                <div className="w-full max-w-3xl relative h-16 mb-4">
                    {/* Road Line */}
                    <div className="absolute bottom-0 w-full h-1 bg-slate-600 rounded-full" />

                    {/* Finish Line Flag */}
                    <div className="absolute right-0 bottom-2 flex flex-col items-center">
                        <Flag className={clsx("w-6 h-6 mb-1 transition-colors", progress === 100 ? "text-green-400 animate-bounce" : "text-slate-500")} />
                        <span className="text-xs text-slate-500 font-mono">GOAL</span>
                    </div>

                    {/* Moving Avatar (Motorcycle) */}
                    <div
                        className="absolute bottom-1 transition-all duration-1000 ease-out flex flex-col items-center"
                        style={{ left: `calc(${animatedProgress}% - 24px)` }}
                    >
                        <div className={clsx("p-2 rounded-full bg-background border border-border shadow-lg transition-transform", progress === 100 && "scale-110 ring-4 ring-yellow-400/50")}>
                            <Bike className={clsx("w-6 h-6", progress === 100 ? "text-yellow-500" : "text-primary")} />
                        </div>
                        {progress === 100 && (
                            <Sparkles className="absolute -top-6 text-yellow-400 w-6 h-6 animate-pulse" />
                        )}
                        <div className="mt-1 bg-slate-800 px-2 py-0.5 rounded text-[10px] text-white font-mono border border-slate-600">
                            {Math.round(animatedProgress)}%
                        </div>
                    </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full max-w-3xl">
                    <ProgressBar current={animatedProgress} total={100} className="h-2 bg-slate-700" />
                    <div className="flex justify-between mt-2 text-xs text-slate-500 font-mono">
                        <span>START</span>
                        <span>{(animatedProgress / 100 * 3).toFixed(1)} / 3 Phases</span>
                        <span>READY</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
