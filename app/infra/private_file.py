"""Restrict local simulation artifacts to their owner and verify the effective DACL."""
import os
import stat
if os.name == "nt":
    import ntsecuritycon
    import win32api
    import win32security

def protect(path):
    if os.name != "nt":
        os.chmod(path, 0o600)
        return
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32security.TOKEN_QUERY)
    try:
        owner=win32security.GetTokenInformation(token,win32security.TokenUser)[0]
    finally:
        token.Close()
    acl=win32security.ACL()
    acl.AddAccessAllowedAce(win32security.ACL_REVISION,ntsecuritycon.FILE_ALL_ACCESS,owner)
    win32security.SetNamedSecurityInfo(str(path),win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,owner,None,acl,None)

def owner_only(path):
    if os.name != "nt":
        return stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    sd=win32security.GetNamedSecurityInfo(str(path),win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION)
    acl=sd.GetSecurityDescriptorDacl()
    if acl is None or acl.GetAceCount()!=1:return False
    flags,mask,sid=acl.GetAce(0)
    return flags[0]==win32security.ACCESS_ALLOWED_ACE_TYPE and sid==sd.GetSecurityDescriptorOwner() and (mask & 3)==3
