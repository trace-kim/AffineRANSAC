; Inno Setup script of MaskReg (docs/SPEC.md D100): dist\MaskReg-setup-<version>.exe from the PyInstaller folder
; dist\MaskReg\. Installs per user (no admin rights) into %LOCALAPPDATA%\Programs\MaskReg, with a Start menu entry
; and an uninstall entry; a newer version installs over the older one.
; Built by packaging/build.py, which passes the version: ISCC packaging\MaskReg.iss /DVersion=X.Y.Z
#ifndef Version
  #error Pass the version: ISCC packaging\MaskReg.iss /DVersion=X.Y.Z (packaging/build.py does)
#endif

[Setup]
; The AppId tells Windows that every version is the same program: never change it.
AppId={{7F19100C-63A7-4790-B42B-DD67E88A1AC8}
AppName=MaskReg
AppVersion={#Version}
AppVerName=MaskReg {#Version}
DefaultDirName={autopf}\MaskReg
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=MaskReg-setup-{#Version}
UninstallDisplayIcon={app}\MaskReg.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[InstallDelete]
; The libraries of the version installed before: files it had and this one has not must not stay behind.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\MaskReg\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Icons]
Name: "{autoprograms}\MaskReg"; Filename: "{app}\MaskReg.exe"
Name: "{autodesktop}\MaskReg"; Filename: "{app}\MaskReg.exe"; Tasks: desktopicon

[Run]
; Not when silent: an update started by MaskReg starts MaskReg again itself (src/affine_ransac/release/update.py).
Filename: "{app}\MaskReg.exe"; Description: "Start MaskReg"; Flags: nowait postinstall skipifsilent
