import re
import tldextract
from urllib.parse import urlparse

def extract_url_features(url):
    parsed = urlparse(url)
    domain_info = tldextract.extract(url)

    features = {
        "url_length": len(url),
        "hostname_length": len(parsed.hostname) if parsed.hostname else 0,
        "path_length": len(parsed.path),
        "has_ip": bool(re.match(r"\d{1,3}(\.\d{1,3}){3}", parsed.hostname or '')),
        "has_at_symbol": "@" in url,
        "count_dots": url.count('.'),
        "count_hyphens": url.count('-'),
        "count_slashes": url.count('/'),
        "uses_https": parsed.scheme == "https",
        "suspicious_words": sum([kw in url.lower() for kw in ["login", "secure", "account", "bank", "verify"]]),
        "domain": domain_info.domain,
        "suffix": domain_info.suffix
    }
    return features

# Example usage
if __name__ == "__main__":
    url = "http://secure-login.bank-example.com/login?user=abc"
    features = extract_url_features(url)
    for k, v in features.items():
        print(f"{k}: {v}")
