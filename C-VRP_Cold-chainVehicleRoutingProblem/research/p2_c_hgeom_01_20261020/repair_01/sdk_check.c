/* Independent Windows SDK layout/snapshot checker. Not compiled or executed in
   this repair: see preflight.json. No hand-copied JOBOBJECT definitions. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stddef.h>

__declspec(dllexport) void sdk_layout(unsigned long long *out) {
    out[0] = sizeof(JOBOBJECT_BASIC_LIMIT_INFORMATION);
    out[1] = sizeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION);
    out[2] = offsetof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, BasicLimitInformation);
    out[3] = offsetof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, IoInfo);
    out[4] = offsetof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, ProcessMemoryLimit);
    out[5] = offsetof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobMemoryLimit);
    out[6] = offsetof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, PeakProcessMemoryUsed);
    out[7] = offsetof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, PeakJobMemoryUsed);
    out[8] = sizeof(SIZE_T);
}

__declspec(dllexport) DWORD sdk_snapshot(HANDLE job, unsigned char *raw,
                                        DWORD capacity, unsigned long long *out) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info;
    DWORD length = 0;
    SIZE_T i;
    volatile unsigned char *bytes = (volatile unsigned char *)&info;
    if (capacity < sizeof(info)) return ERROR_INSUFFICIENT_BUFFER;
    for (i = 0; i < sizeof(info); ++i) bytes[i] = 0;
    if (!QueryInformationJobObject(job, JobObjectExtendedLimitInformation,
                                   &info, sizeof(info), &length))
        return GetLastError();
    for (i = 0; i < sizeof(info); ++i) raw[i] = bytes[i];
    out[0] = info.BasicLimitInformation.LimitFlags;
    out[1] = info.BasicLimitInformation.ActiveProcessLimit;
    out[2] = info.ProcessMemoryLimit;
    out[3] = info.JobMemoryLimit;
    out[4] = info.PeakProcessMemoryUsed;
    out[5] = info.PeakJobMemoryUsed;
    out[6] = length;
    return ERROR_SUCCESS;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance; (void)reason; (void)reserved;
    return TRUE;
}
