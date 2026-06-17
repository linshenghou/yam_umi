<#
.SYNOPSIS
  Open the Windows firewall so a LAN peer (e.g. Ubuntu) can reach the ZMQ
  Vive-tracker pose stream from sim_teleop.stream_pose, and reply to ping.

.DESCRIPTION
  Does three things, all idempotent:
    1. Flips every "Public" network interface to "Private" (trusted LAN).
  Public blocks almost all inbound traffic; Private is correct for a lab/
  workshop network.
    2. Allows inbound TCP <Port> (default 1234) on Private profiles — this is
  what the ZMQ PUB socket actually needs (Windows binds, Ubuntu connects in).
    3. Enables the inbound ICMPv4 Echo Request rule so `ping <WIN_IP>` works
  for diagnostics.

  Self-elevates: run from a normal shell and approve the UAC prompt.

.PARAMETER Port
  TCP port the publisher binds (default 1234). Must match stream_pose --port.

.EXAMPLE
  .\scripts\open_stream_firewall.ps1
  .\scripts\open_stream_firewall.ps1 -Port 5555
#>
param(
    [int]$Port = 1234,
    [string]$RuleName = "ZMQ pose stream $Port"
)

# --- Self-elevate -----------------------------------------------------------
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$currentUser).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    Start-Process -FilePath "powershell.exe" `
        -Verb RunAs `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Port", $Port
    exit
}

# --- 1. Public -> Private ---------------------------------------------------
$public = Get-NetConnectionProfile | Where-Object { $_.NetworkCategory -eq 'Public' }
if ($public) {
    $public | Set-NetConnectionProfile -NetworkCategory Private
    Write-Host "[OK] Set interface(s) to Private:" -ForegroundColor Green
    $public | ForEach-Object { Write-Host "      $($_.Name) ($($_.InterfaceAlias))" }
} else {
    Write-Host "[SKIP] No Public interfaces; nothing to flip." -ForegroundColor DarkGray
}

# --- 2. Inbound TCP <Port> for ZMQ PUB -------------------------------------
$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[SKIP] Firewall rule '$RuleName' already exists." -ForegroundColor DarkGray
} else {
    New-NetFirewallRule `
        -DisplayName $RuleName `
        -Group "sim_teleop" `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort $Port `
        -Action Allow `
        -Profile Private `
        -Enabled True | Out-Null
    Write-Host "[OK] Allowed inbound TCP $Port (Private) as '$RuleName'." -ForegroundColor Green
}

# --- 3. Allow inbound ping (ICMPv4 echo) -----------------------------------
# A dedicated rule is more reliable than enabling the built-in "File and
# Printer Sharing" rules, whose Enabled enum field doesn't toggle cleanly.
$icmpName = "Allow Ping (ICMPv4-In)"
if (Get-NetFirewallRule -DisplayName $icmpName -ErrorAction SilentlyContinue) {
    Write-Host "[SKIP] Firewall rule '$icmpName' already exists." -ForegroundColor DarkGray
} else {
    New-NetFirewallRule `
        -DisplayName $icmpName `
        -Group "sim_teleop" `
        -Direction Inbound `
        -Action Allow `
        -Protocol ICMPv4 `
        -IcmpType 8 `
        -Profile Private `
        -Enabled True | Out-Null
    Write-Host "[OK] Allowed inbound ICMPv4 echo request (ping) on Private." -ForegroundColor Green
}

# --- Summary ----------------------------------------------------------------
Write-Host ""
Write-Host "Done. Current connection profile:" -ForegroundColor Cyan
Get-NetConnectionProfile | Select-Object Name, InterfaceAlias, NetworkCategory | Format-Table -AutoSize
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -notmatch 'Loopback|vEthernet|WSL' } |
    Sort-Object InterfaceAlias)
Write-Host "Your LAN IPv4 address(es) — give one to the Ubuntu receiver:" -ForegroundColor Cyan
$ip | ForEach-Object { Write-Host "      $($_.InterfaceAlias): $($_.IPAddress)" }
Write-Host ""
Write-Host "On Ubuntu:" -ForegroundColor Cyan
Write-Host "    ping <WIN_IP>" -ForegroundColor White
Write-Host "    python -m sim_teleop.receive_pose --host <WIN_IP> --port $Port" -ForegroundColor White
Write-Host ""
Write-Host "Press Enter to close this window..." -ForegroundColor DarkGray
Read-Host | Out-Null
