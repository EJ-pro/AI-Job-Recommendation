import { NextResponse } from 'next/server';
import { saveTestResult, getTestResults, TestResult } from '@/lib/storage';

export async function POST(request: Request) {
    try {
        const body = await request.json();
        const { userId, jobRole, difficulty, scores } = body;

        if (!jobRole || !difficulty) {
            return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
        }

        const newResult: TestResult = {
            id: crypto.randomUUID(), // Use native crypto
            userId: userId || 'anonymous',
            jobRole,
            scores: scores || {}, // Default to empty if not provided
            difficulty,
            timestamp: new Date().toISOString(),
        };

        await saveTestResult(newResult);

        return NextResponse.json({ success: true, result: newResult });
    } catch (error) {
        console.error('Failed to save result:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const userId = searchParams.get('userId');

        let allResults: TestResult[] = [];
        try {
            allResults = await getTestResults();
        } catch (e) {
            // ignore if no results yet
        }

        const userResults = userId
            ? allResults.filter(r => r.userId === userId)
            : allResults;

        return NextResponse.json({ success: true, results: userResults });
    } catch (error) {
        console.error('Failed to fetch results:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
