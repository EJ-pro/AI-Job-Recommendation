import fs from 'fs/promises';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'data');
const DATA_FILE = path.join(DATA_DIR, 'results.json');

export interface TestResult {
    id: string; // uuid
    userId?: string; // email or name if logged in
    jobRole: string;
    scores: Record<string, number>;
    difficulty: 'beginner' | 'advanced';
    timestamp: string;
}

async function ensureDataDir() {
    try {
        await fs.access(DATA_DIR);
    } catch {
        await fs.mkdir(DATA_DIR, { recursive: true });
    }
}

export async function saveTestResult(result: TestResult): Promise<void> {
    await ensureDataDir();

    let currentData: TestResult[] = [];
    try {
        const fileContent = await fs.readFile(DATA_FILE, 'utf-8');
        currentData = JSON.parse(fileContent);
    } catch (error) {
        // File might not exist or be empty, ignore
    }

    currentData.push(result);
    await fs.writeFile(DATA_FILE, JSON.stringify(currentData, null, 2));
}

export async function getTestResults(): Promise<TestResult[]> {
    try {
        const fileContent = await fs.readFile(DATA_FILE, 'utf-8');
        return JSON.parse(fileContent);
    } catch (error) {
        return [];
    }
}
