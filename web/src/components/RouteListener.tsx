'use client';

import { usePathname } from 'next/navigation';

export default function RouteListener() {
    const pathname = usePathname();
    const hideAds = pathname?.startsWith('/login') || pathname?.startsWith('/mypage') || pathname?.startsWith('/test');

    return (
        <div data-hide-ads={hideAds} className="hidden" />
    );
}
