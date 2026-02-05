import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

interface ProgressBarProps {
    current: number;
    total: number;
    className?: string;
}

export default function ProgressBar({ current, total, className }: ProgressBarProps) {
    const progress = Math.min(100, Math.max(0, (current / total) * 100));

    return (
        <div className={twMerge('w-full bg-muted rounded-full h-2.5 overflow-hidden', className)}>
            <div
                className="h-full bg-primary transition-all duration-500 ease-out rounded-full shadow-[0_0_10px_rgba(79,70,229,0.5)]"
                style={{ width: `${progress}%` }}
            />
        </div>
    );
}
