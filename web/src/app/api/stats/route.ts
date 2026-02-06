import { NextResponse } from 'next/server';
import { getTestResults } from '@/lib/storage';

export async function GET() {
    try {
        const results = await getTestResults();

        // Aggregate statistics
        const jobCounts: Record<string, number> = {};
        const difficultyCounts = { beginner: 0, advanced: 0 };

        results.forEach(r => {
            // Job Counts
            jobCounts[r.jobRole] = (jobCounts[r.jobRole] || 0) + 1;

            // Difficulty Counts
            if (r.difficulty === 'beginner') difficultyCounts.beginner++;
            else if (r.difficulty === 'advanced') difficultyCounts.advanced++;
        });

        const total = results.length;

        return NextResponse.json({
            total,
            jobCounts,
            difficultyCounts
        });
    } catch (error) {
        console.error('Failed to get stats:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
