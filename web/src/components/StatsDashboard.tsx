'use client';

import { useEffect, useState } from 'react';

interface Stats {
    total: number;
    jobCounts: Record<string, number>;
    difficultyCounts: {
        beginner: number;
        advanced: number;
    };
}

export default function StatsDashboard() {
    const [stats, setStats] = useState<Stats | null>(null);

    useEffect(() => {
        fetch('/api/stats')
            .then(res => res.json())
            .then(data => setStats(data))
            .catch(err => console.error('Failed to fetch stats:', err));
    }, []);

    if (!stats) return <div className="p-4 text-center">데이터 로딩 중...</div>;

    // Calculate percentages for jobs
    const sortedJobs = Object.entries(stats.jobCounts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5); // Top 5

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4">

            {/* Difficulty Distribution (Pie/Bar) */}
            <div className="bg-white/50 dark:bg-zinc-900/50 rounded-2xl p-6 border border-border">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    👥 사용자 분포
                    <span className="text-xs font-normal text-muted-foreground">
                        (입문자 vs 실전)
                    </span>
                </h3>
                <div className="flex items-center gap-4 h-32">
                    <div className="flex-1 space-y-2">
                        <div className="flex justify-between text-sm">
                            <span>입문자 (Beginner)</span>
                            <span className="font-bold">{stats.difficultyCounts.beginner}명</span>
                        </div>
                        <div className="h-3 bg-secondary/20 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500"
                                style={{ width: `${(stats.difficultyCounts.beginner / stats.total) * 100}%` }}
                            />
                        </div>

                        <div className="pt-2 flex justify-between text-sm">
                            <span>실전 (Advanced)</span>
                            <span className="font-bold">{stats.difficultyCounts.advanced}명</span>
                        </div>
                        <div className="h-3 bg-secondary/20 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-purple-500"
                                style={{ width: `${(stats.difficultyCounts.advanced / stats.total) * 100}%` }}
                            />
                        </div>
                    </div>
                </div>
                <p className="text-right text-xs text-muted-foreground mt-2">
                    총 참여자: {stats.total}명
                </p>
            </div>

            {/* Job Popularity (Bar Chart) */}
            <div className="bg-white/50 dark:bg-zinc-900/50 rounded-2xl p-6 border border-border">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                    🏆 인기 직무 TOP 5
                </h3>
                <div className="space-y-3">
                    {sortedJobs.map(([jobId, count], idx) => (
                        <div key={jobId} className="space-y-1">
                            <div className="flex justify-between text-xs font-medium">
                                <span className="uppercase">{jobId}</span>
                                <span>{count}명</span>
                            </div>
                            <div className="h-2 bg-secondary/20 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-primary to-primary/60"
                                    style={{ width: `${(count / stats.total) * 100}%` }}
                                />
                            </div>
                        </div>
                    ))}
                    {sortedJobs.length === 0 && (
                        <p className="text-sm text-muted-foreground text-center py-4">아직 데이터가 없습니다.</p>
                    )}
                </div>
            </div>
        </div>
    );
}
