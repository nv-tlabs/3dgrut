#!/bin/bash
# Check ABI compatibility between a Kaolin wheel and a PyTorch wheel.
#
# Usage:
#   ./scripts/test_kaolin_pytorch_abi.sh [options]
#
# Options:
#   --kaolin-find-link URL   Kaolin find-links page (required)
#   --torch-index URL        PyTorch index base URL, e.g. https://download.pytorch.org/whl/cu128 (required)
#   --torch-version VER      PyTorch version to match, e.g. 2.8.0 (required)
#   --python VER             Python version, e.g. 3.11 (default: current interpreter)
#
# Examples:
#   ./scripts/test_kaolin_pytorch_abi.sh \
#       --kaolin-find-link https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html \
#       --torch-index https://download.pytorch.org/whl/cu128 \
#       --torch-version 2.8.0
#
#   ./scripts/test_kaolin_pytorch_abi.sh \
#       --kaolin-find-link https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html \
#       --torch-index https://download.pytorch.org/whl/cu126 \
#       --torch-version 2.6.0 \
#       --python 3.11

set -euo pipefail

KAOLIN_FIND_LINK=""
TORCH_INDEX=""
TORCH_VERSION=""
PYVER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kaolin-find-link) KAOLIN_FIND_LINK="$2"; shift 2 ;;
        --torch-index)      TORCH_INDEX="$2"; shift 2 ;;
        --torch-version)    TORCH_VERSION="$2"; shift 2 ;;
        --python)           PYVER="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$KAOLIN_FIND_LINK" ]] && { echo "ERROR: --kaolin-find-link is required" >&2; exit 1; }
[[ -z "$TORCH_INDEX" ]]     && { echo "ERROR: --torch-index is required" >&2; exit 1; }
[[ -z "$TORCH_VERSION" ]]   && { echo "ERROR: --torch-version is required" >&2; exit 1; }

PYVER="${PYVER:-$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')}"
CPVER="cp${PYVER/./}"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ── helpers ──────────────────────────────────────────────────────────

# Parse href from an HTML page (handles single and double quotes).
# Matches against the URL-decoded form but returns the original href.
find_wheel_url() {
    local page_url="$1" name_pattern="$2"

    local page
    page=$(curl -fsSL "$page_url")

    # Extract all hrefs, then match decoded filenames against the pattern
    local href
    href=$(echo "$page" \
        | grep -oP "href=['\"]?\K[^'\"# >]+" \
        | while IFS= read -r raw; do
            decoded=$(python3 -c "import sys,urllib.parse; print(urllib.parse.unquote(sys.stdin.read().strip()))" <<< "$raw")
            if echo "$decoded" | grep -qiE "${name_pattern}"; then
                echo "$raw"
                break
            fi
        done)

    [[ -z "$href" ]] && return 1

    # Resolve relative URLs
    if [[ ! "$href" =~ ^https?:// ]]; then
        local origin
        origin=$(echo "$page_url" | grep -oP '^https?://[^/]+')
        if [[ "$href" =~ ^/ ]]; then
            href="${origin}${href}"
        else
            local base
            base=$(echo "$page_url" | sed 's|[^/]*$||')
            href="${base}${href}"
        fi
    fi

    echo "$href"
}

download_wheel() {
    local url="$1" dest_dir="$2"
    local fname
    fname=$(basename "$url" | sed 's/#.*//' | python3 -c "import sys,urllib.parse; print(urllib.parse.unquote(sys.stdin.read().strip()))")
    echo "  Downloading: $fname" >&2
    curl -fsSL -o "$dest_dir/$fname" "$url"
    echo "$dest_dir/$fname"
}

extract_so() {
    local whl="$1" glob_pattern="$2" dest_dir="$3"
    local so_path
    so_path=$(unzip -l "$whl" | awk '{print $NF}' | grep -E "$glob_pattern" | head -1)
    [[ -z "$so_path" ]] && { echo "  ERROR: No file matching '$glob_pattern' in $(basename "$whl")" >&2; return 1; }
    unzip -qo "$whl" "$so_path" -d "$dest_dir"
    echo "$dest_dir/$so_path"
}

# ── download wheels ──────────────────────────────────────────────────
echo "=== Downloading wheels (python ${PYVER}) ==="
echo ""

echo "Kaolin:"
echo "  Index: $KAOLIN_FIND_LINK"
kaolin_url=$(find_wheel_url "$KAOLIN_FIND_LINK" "kaolin-.*-${CPVER}-${CPVER}-(linux_x86_64|manylinux).*\\.whl") \
    || { echo "  ERROR: No Kaolin wheel for ${CPVER} linux_x86_64 found" >&2; exit 1; }
kaolin_whl=$(download_wheel "$kaolin_url" "$WORK")
echo ""

echo "PyTorch (torch ${TORCH_VERSION}):"
torch_page="${TORCH_INDEX%/}/torch/"
echo "  Index: $torch_page"
# Escape dots in version for regex, match url-encoded + (%2B)
torch_ver_re="${TORCH_VERSION//./\\.}"
torch_url=$(find_wheel_url "$torch_page" "torch-${torch_ver_re}(%2B|\\+)[^/]*-${CPVER}-${CPVER}-(linux_x86_64|manylinux).*\\.whl") \
    || { echo "  ERROR: No torch ${TORCH_VERSION} wheel for ${CPVER} linux_x86_64 found" >&2; exit 1; }
torch_whl=$(download_wheel "$torch_url" "$WORK")
echo ""

# ── extract shared objects ───────────────────────────────────────────
echo "=== Extracting shared objects ==="
kaolin_so=$(extract_so "$kaolin_whl" '_C.*\.so' "$WORK/kaolin")
echo "  Kaolin:  $(basename "$kaolin_so")"

torch_c10=$(extract_so "$torch_whl" 'libc10\.so' "$WORK/torch")
echo "  libc10:  $(basename "$torch_c10")"

torch_libs=()
for lib in libtorch_cpu libtorch_python libc10_cuda libtorch_cuda libtorch; do
    local_path=$(extract_so "$torch_whl" "${lib}\\.so" "$WORK/torch" 2>/dev/null) && {
        echo "  ${lib}: found"
        torch_libs+=("$local_path")
    } || true
done
echo ""

# ── collect symbols ──────────────────────────────────────────────────
echo "=== Checking ABI compatibility ==="
echo ""

# Undefined symbols kaolin needs
nm -D "$kaolin_so" 2>/dev/null | awk '$1 == "U" {print $2}' | sort -u > "$WORK/kaolin_undefined.txt"

# Defined symbols torch provides (T=text, W=weak, V=vtable, B=bss)
{
    nm -D "$torch_c10" 2>/dev/null
    for lib in "${torch_libs[@]}"; do
        nm -D "$lib" 2>/dev/null
    done
} | awk '$2 ~ /^[TWVBD]$/ {print $3}' | sort -u > "$WORK/torch_defined.txt"

# Filter to c10/at/torch namespaces only
grep -E '^_ZN(3c10|2at|5torch)' "$WORK/kaolin_undefined.txt" > "$WORK/kaolin_needs.txt" || true

missing=$(comm -23 "$WORK/kaolin_needs.txt" "$WORK/torch_defined.txt")

# ── ABI detection ────────────────────────────────────────────────────
# Ss = old ABI std::string; NSt7__cxx11 = new ABI std::__cxx11::basic_string
detect_abi() {
    local file="$1" label="$2"
    local has_old has_new
    has_old=$(grep -c 'Ss' "$file" || true)
    has_new=$(grep -c 'NSt7__cxx11' "$file" || true)
    if [[ "$has_old" -gt 0 && "$has_new" -eq 0 ]]; then
        echo "$label ABI: old (_GLIBCXX_USE_CXX11_ABI=0)"
    elif [[ "$has_new" -gt 0 && "$has_old" -eq 0 ]]; then
        echo "$label ABI: new (_GLIBCXX_USE_CXX11_ABI=1)"
    elif [[ "$has_new" -gt 0 && "$has_old" -gt 0 ]]; then
        echo "$label ABI: mixed (old=$has_old, new=$has_new)"
    else
        echo "$label ABI: unknown (no std::string symbols)"
    fi
}

detect_abi "$WORK/kaolin_needs.txt" "Kaolin"
detect_abi "$WORK/torch_defined.txt" "Torch "
echo ""

# ── report ───────────────────────────────────────────────────────────
kaolin_needs_count=$(wc -l < "$WORK/kaolin_needs.txt")
missing_count=$(echo "$missing" | grep -c . || true)

echo "Kaolin requires $kaolin_needs_count c10/at/torch symbols"
echo "Missing symbols: $missing_count"
echo ""

if [[ "$missing_count" -eq 0 ]]; then
    echo "PASS: All symbols resolved. Wheels are ABI-compatible."
    exit 0
else
    echo "FAIL: $missing_count symbols unresolved — ABI mismatch."
    echo ""
    echo "Missing symbols (demangled):"
    echo "$missing" | head -30 | while IFS= read -r sym; do
        demangled=$(c++filt "$sym" 2>/dev/null || echo "$sym")
        printf "  %-80s\n    → %s\n" "$sym" "$demangled"
    done
    [[ "$missing_count" -gt 30 ]] && echo "  ... and $((missing_count - 30)) more"
    exit 1
fi
