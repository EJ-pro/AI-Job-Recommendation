'use client';

import { useEffect, useState } from 'react';
import { JOBS } from '@/lib/data';
import JobScoreChart from '@/components/JobScoreChart';
import { Check, Loader2 } from 'lucide-react';

interface TestResult {
    jobRole: string;
    scores?: Record<string, number>;
}

export default function PersonalResultSummary({ userEmail }: { userEmail: string }) {
    const [results, setResults] = useState<TestResult[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!userEmail) return;

        fetch(`/api/test-result?userId=${userEmail}`)
            .then((res) => res.json())
            .then((data) => {
                if (data.success) {
                    setResults(data.results);
                }
                setLoading(false);
            })
            .catch((err) => {
                console.error(err);
                setLoading(false);
            });
    }, [userEmail]);

    if (loading) return <div className="flex justify-center p-8"><Loader2 className="animate-spin text-primary" /></div>;
    if (results.length === 0) return <div className="text-center p-8 text-muted-foreground">아직 진행한 테스트 결과가 없습니다.</div>;

    // Calculate Frequencies
    const jobCounts: Record<string, number> = {};
    results.forEach((r) => {
        jobCounts[r.jobRole] = (jobCounts[r.jobRole] || 0) + 1;
    });

    const sortedJobs = Object.entries(jobCounts)
        .sort(([, a], [, b]) => b - a)
        .map(([id, count]) => ({
            id,
            name: JOBS.find((j) => j.id === id)?.title || id,
            count
        }));

    const topJobId = sortedJobs[0]?.id;
    const topJob = JOBS.find((j) => j.id === topJobId);
    const top2JobId = sortedJobs[1]?.id;
    const top2Job = JOBS.find((j) => j.id === top2JobId);

    // Prepare text
    const mainText = top2Job
        ? `${topJob?.title}와 ${top2Job?.title}`
        : topJob?.title;

    // Use tags or description for "Common Traits"
    const traits = topJob?.tags || ['분석적 사고', '논리적 구조화', '기술적 호기심'];

    return (
        <div className="bg-card border border-border rounded-3xl p-8 shadow-sm space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="text-center space-y-4">
                <p className="text-muted-foreground">여러 차례 테스트를 진행한 결과,</p>
                <h3 className="text-2xl md:text-3xl font-bold leading-tight">
                    <span className="text-primary">{mainText}</span> 직무가 <br className="md:hidden" />
                    가장 높은 비중으로 반복적으로 도출되었습니다.
                </h3>
            </div>

            <div className="bg-muted/30 rounded-2xl p-6 md:p-8">
                <div className="flex flex-wrap gap-4 justify-center mb-8">
                    {topJob && (
                        <div className="px-4 py-2 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 font-bold rounded-lg flex items-center gap-2">
                            🥇 {topJob.title}
                        </div>
                    )}
                    {top2Job && (
                        <div className="px-4 py-2 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 font-bold rounded-lg flex items-center gap-2">
                            🥈 {top2Job.title}
                        </div>
                    )}
                </div>

                <JobScoreChart data={sortedJobs.slice(0, 5)} />
            </div>

            <div className="space-y-4 max-w-2xl mx-auto">
                <h4 className="text-lg font-bold text-center">공통적으로 나타난 특징은</h4>
                <ul className="space-y-3">
                    {traits.map((trait, i) => (
                        <li key={i} className="flex items-center gap-3 text-lg">
                            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-yellow-400 flex items-center justify-center text-white text-xs">
                                <Check size={14} strokeWidth={4} />
                            </div>
                            <span>{trait} 성향</span>
                        </li>
                    ))}
                    <li className="flex items-center gap-3 text-lg">
                        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-yellow-400 flex items-center justify-center text-white text-xs">
                            <Check size={14} strokeWidth={4} />
                        </div>
                        <span>실습과 결과 검증 중심의 학습 성향이었다.</span>
                    </li>
                </ul>
            </div>

            <div className="bg-muted/50 p-6 rounded-xl text-center text-muted-foreground leading-relaxed">
                이는 사용자가 단순한 이론 학습보다는 <br className="md:hidden" />
                <span className="font-bold text-foreground">실제 데이터를 다루고 결과를 만들어내는 역할</span>에 더 <br className="md:hidden" />
                적합한 성향을 가지고 있음을 보여줍니다.
            </div>
        </div>
    );
}
