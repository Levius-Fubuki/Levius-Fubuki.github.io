#!/usr/bin/env python3
"""Rebuild the homepage's local search data from Hexo's existing search.xml."""
import json
from html.parser import HTMLParser
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def main():
    items = []
    for entry in ET.parse(ROOT / 'search.xml').getroot().findall('entry'):
        parser = PlainText()
        parser.feed(entry.findtext('content', ''))
        items.append({
            'title': entry.findtext('title'),
            'url': entry.findtext('url'),
            'text': ' '.join(' '.join(parser.parts).split()),
        })
    (ROOT / 'js/search-index.json').write_text(
        json.dumps(items, ensure_ascii=False, separators=(',', ':')), encoding='utf-8'
    )
    print(f'Indexed {len(items)} articles.')


if __name__ == '__main__':
    main()
