'use client';

import { useEffect } from 'react';

interface AdBannerProps {
    dataAdSlot: string;
    dataAdFormat?: string;
    dataFullWidthResponsive?: boolean;
    className?: string;
}

export default function AdBanner({
    dataAdSlot,
    dataAdFormat = 'auto',
    dataFullWidthResponsive = true,
    className,
}: AdBannerProps) {
    useEffect(() => {
        try {
            if (process.env.NODE_ENV === 'production') {
                // @ts-ignore
                (window.adsbygoogle = window.adsbygoogle || []).push({});
            }
        } catch (err) {
            console.error('AdSense error:', err);
        }
    }, []);

    // Development Fallback
    if (process.env.NODE_ENV !== 'production') {
        return (
            <div
                className={`bg-muted/30 border-2 border-dashed border-muted-foreground/20 rounded-lg flex items-center justify-center text-muted-foreground text-sm font-medium ${className}`}
                style={{ minHeight: '250px', width: '100%' }}
            >
                Google AdSense<br />
                (Slot: {dataAdSlot})
            </div>
        );
    }

    return (
        <div className={className}>
            <ins
                className="adsbygoogle"
                style={{ display: 'block' }}
                data-ad-client="ca-pub-6922660715404171" // User provided ID
                data-ad-slot={dataAdSlot}
                data-ad-format={dataAdFormat}
                data-full-width-responsive={dataFullWidthResponsive.toString()}
            />
        </div>
    );
}
