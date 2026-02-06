import { NextResponse } from 'next/server';
import { saveTestResult } from '@/lib/storage';
// import { v4 as uuidv4 } from 'uuid'; // Removed uuid dependency

export async function POST(request: Request) {
    try {
        const body = await request.json();
        const { userId, jobRole, difficulty } = body;

        if (!jobRole || !difficulty) {
            return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
        }

        const newResult = {
            id: crypto.randomUUID(), // Use native crypto
            userId: userId || 'anonymous',
            jobRole,
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
