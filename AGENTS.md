# Browser CLI - Agent Instructions

This document provides complete instructions for AI agents (LLMs) on how to use the browser-cli tool for web automation tasks.

## Overview

browser-cli is a command-line tool that controls Chrome via the Chrome DevTools Protocol. It allows you to programmatically control a browser, navigate pages, extract content, and interact with web elements.

## Critical Prerequisites

**BEFORE using ANY browser command, you MUST start the browser first:**

```bash
browser start
```

This starts Chrome with remote debugging enabled on port 9222. All other commands will fail if the browser is not running.

## Command Reference

### 1. Starting the Browser

**Command:** `browser start [--profile]`

**When to use:**
- At the beginning of ANY workflow involving browser automation
- After system restart
- If you get "Could not connect to browser" errors

**Options:**
- No flags: Fresh browser profile (no cookies, no login sessions)
- `--profile`: Use existing Chrome profile (preserves logins, cookies, history)

**Examples:**
```bash
# Start with fresh profile
browser start

# Start with your existing Chrome profile (use for sites requiring login)
browser start --profile
```

**Important:**
- Only ONE browser instance can run at a time
- The browser runs in the background (you can continue using other commands)
- The browser window will open visibly - this is expected behavior

---

### 2. Navigation

**Command:** `browser nav <url> [--new]`

**When to use:**
- To navigate to any URL
- To open a new tab

**Parameters:**
- `<url>`: Full URL including protocol (http:// or https://)
- `--new`: (Optional) Open in new tab instead of current tab

**Examples:**
```bash
# Navigate in current tab
browser nav https://example.com

# Open in new tab
browser nav https://example.com --new
```

**Important:**
- Always include the protocol (https://)
- Command waits for DOM content to load before returning
- Use `--new` if you need to preserve the current tab

---

### 3. JavaScript Evaluation

**Command:** `browser eval '<javascript-code>'`

**When to use:**
- Extract data from the page
- Inspect page structure
- Count elements
- Get page metadata
- Execute any JavaScript in the browser context

**The code runs in an async context, so you can use await.**

**Examples:**

```bash
# Get page title
browser eval "document.title"

# Count all links
browser eval "document.querySelectorAll('a').length"

# Extract all headings
browser eval "Array.from(document.querySelectorAll('h1, h2, h3')).map(h => ({tag: h.tagName, text: h.textContent.trim()}))"

# Get all links with text
browser eval "Array.from(document.querySelectorAll('a')).map(a => ({href: a.href, text: a.textContent.trim()})).filter(link => link.text.length > 0)"

# Check if element exists
browser eval "document.querySelector('.login-button') !== null"

# Get form data
browser eval "Object.fromEntries(new FormData(document.querySelector('form')))"

# Extract table data
browser eval "Array.from(document.querySelectorAll('table tr')).map(row => Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim()))"

# Get meta tags
browser eval "Array.from(document.querySelectorAll('meta')).map(m => ({name: m.name || m.property, content: m.content}))"

# Check if user is logged in (example)
browser eval "document.querySelector('.user-avatar') !== null"
```

**Output formatting:**
- Arrays of objects: Each object is printed with key-value pairs
- Single objects: Printed as key-value pairs
- Primitives: Printed directly

**Important:**
- Always wrap code in quotes
- Use single quotes for JS strings inside double-quoted bash strings
- The code executes in the page context, not Node.js context
- DOM APIs are available (document, window, etc.)

---

### 4. Screenshots

**Command:** `browser screenshot`

**When to use:**
- Visual verification of page state
- Debugging layout issues
- Capturing evidence of page content
- Before/after comparisons

**Examples:**
```bash
browser screenshot
# Output: /tmp/screenshot-2024-01-15T10-30-45-123Z.png
```

**Output:**
- Prints the full path to the saved PNG file
- File is saved in system temp directory
- Filename includes timestamp for uniqueness

**Important:**
- Captures only the visible viewport (not full page)
- Screenshot is saved immediately
- File persists after command completes

---

### 5. Cookie Inspection

**Command:** `browser cookies`

**When to use:**
- Debug authentication issues
- Verify session state
- Check cookie security flags
- Understand cookie scope

**Examples:**
```bash
browser cookies
```

**Output format:**
```
session_id: abc123xyz
  domain: .example.com
  path: /
  httpOnly: true
  secure: true

user_pref: dark_mode
  domain: example.com
  path: /
  httpOnly: false
  secure: false
```

**Important:**
- Shows cookies for the current page's domain
- Includes security flags (httpOnly, secure)
- Shows domain scope and path

---

### 6. Content Extraction

**Command:** `browser content [url]`

**When to use:**
- Extract readable article content
- Convert web pages to markdown
- Get clean text without navigation/ads
- Archive article content

**Parameters:**
- `[url]`: (Optional) URL to navigate to before extracting

**Examples:**
```bash
# Extract content from current page
browser content

# Navigate and extract in one command
browser content https://example.com/article

# Save to file
browser content https://example.com/article > article.md
```

**How it works:**
1. Uses Mozilla Readability to identify main content
2. Removes navigation, ads, footers, etc.
3. Converts HTML to clean Markdown
4. Falls back to main content area if Readability fails

**Output format:**
```
URL: https://example.com/article

# Article Title

Main content in markdown format...
```

**Important:**
- Works best on article-style content
- May not work well on complex web applications
- 30-second timeout for extraction
- Use `> filename.md` to save to file

---

### 7. Google Search

**Command:** `browser search "<query>" [-n <num>] [--content]`

**When to use:**
- Research topics
- Find relevant URLs
- Gather information from multiple sources
- Content aggregation

**Parameters:**
- `<query>`: Search query (use quotes if multi-word)
- `-n <num>`: Number of results (default: 5, max: ~100)
- `--content`: Also fetch and extract readable content from each result

**Examples:**
```bash
# Basic search (5 results)
browser search "typescript best practices"

# Get 10 results
browser search "rust programming" -n 10

# Search and fetch content from first 3 results
browser search "climate change solutions" -n 3 --content

# Save results to file
browser search "machine learning tutorials" -n 5 --content > research.txt
```

**Output format:**
```
--- Result 1 ---
Title: TypeScript Best Practices 2024
Link: https://example.com/typescript-guide
Snippet: Learn the best practices for TypeScript development...
Content:
[Markdown content if --content flag used]

--- Result 2 ---
...
```

**Important:**
- Uses Google search (requires internet connection)
- With `--content`: Much slower, fetches and processes each result
- Without `--content`: Fast, only returns title/link/snippet
- Automatically handles pagination
- Removes duplicate results
- 60-second timeout for search operations

---

### 8. Interactive Element Picker

**Command:** `browser pick "<message>"`

**When to use:**
- Identify CSS selectors for elements
- Understand page structure
- Debug element selection issues
- Get element metadata

**Parameters:**
- `<message>`: Instructions to display to the user (if human is present)

**How it works:**
1. Overlays the page with an interactive picker
2. Hover over elements to highlight them
3. Click to select (or Cmd/Ctrl+click for multiple)
4. Press Enter to finish (if multiple selected)
5. Press ESC to cancel

**Examples:**
```bash
browser pick "Select the login button"
browser pick "Find the main navigation menu"
browser pick "Select all product cards"
```

**Output format:**
```
tag: button
id: login-btn
class: btn btn-primary
text: Login to your account
html: <button id="login-btn" class="btn btn-primary">Login to your account</button>
parents: div.auth-container > form#login-form > div.form-actions
```

**Important:**
- This is INTERACTIVE - requires a human to click elements
- DO NOT use in fully automated workflows
- Use for discovery/debugging, not production automation
- The browser window must be visible and focused

---

## Common Workflows

### Workflow 1: Extract Article Content

```bash
# Start browser
browser start

# Navigate to article
browser nav https://example.com/article

# Extract content to file
browser content > article.md

# Optional: Take screenshot for reference
browser screenshot
```

### Workflow 2: Research Topic

```bash
# Start with profile (if Google login needed)
browser start --profile

# Search and fetch content from top 5 results
browser search "topic name" -n 5 --content > research.txt
```

### Workflow 3: Data Extraction

```bash
# Start browser
browser start

# Navigate to target page
browser nav https://example.com/data

# Extract specific data
browser eval "Array.from(document.querySelectorAll('.data-item')).map(item => ({
  title: item.querySelector('.title').textContent,
  value: item.querySelector('.value').textContent,
  link: item.querySelector('a').href
}))" > data.json
```

### Workflow 4: Multi-Page Scraping

```bash
# Start browser
browser start

# Navigate to first page
browser nav https://example.com/page/1

# Extract data
browser eval "/* extraction code */" > page1.json

# Navigate to next page
browser nav https://example.com/page/2

# Extract data
browser eval "/* extraction code */" > page2.json
```

### Workflow 5: Authentication Check

```bash
# Start with profile to use saved cookies
browser start --profile

# Navigate to protected page
browser nav https://example.com/dashboard

# Check if logged in
browser eval "document.querySelector('.user-menu') !== null"

# If true, user is logged in; if false, need to login
```

---

## Error Handling

### Error: "Could not connect to browser"

**Cause:** Browser is not running

**Solution:**
```bash
browser start
```

### Error: "No active tab found"

**Cause:** Browser has no open tabs

**Solution:**
```bash
browser nav https://google.com
```

### Error: "Timeout after Xs"

**Cause:** Operation took too long (network issue, slow page, etc.)

**Solution:**
- Check internet connection
- Try again with a simpler page
- Increase timeout in source code if needed

### Error: Command not found

**Cause:** browser-cli not installed globally

**Solution:**
```bash
cd ~/projects/browser-cli
npm link
```

---

## Best Practices for AI Agents

### 1. Always Start the Browser First

```bash
# ✅ CORRECT
browser start
browser nav https://example.com

# ❌ WRONG - will fail
browser nav https://example.com
```

### 2. Chain Commands Sequentially

Each command should complete before the next begins:

```bash
# ✅ CORRECT - separate bash calls
browser start
browser nav https://example.com
browser content > article.md

# ❌ WRONG - commands will run in parallel
browser start && browser nav https://example.com && browser content
```

### 3. Use Proper Quoting

```bash
# ✅ CORRECT
browser eval "document.title"
browser search "machine learning"

# ❌ WRONG
browser eval document.title
browser search machine learning
```

### 4. Save Output When Needed

```bash
# ✅ CORRECT - save to file
browser content > article.md
browser search "topic" -n 5 --content > research.txt

# ❌ WRONG - output only to terminal
browser content
```

### 5. Handle Long-Running Operations

For searches with `--content`, expect delays:

```bash
# This will take time (fetches 10 pages)
browser search "topic" -n 10 --content

# Better: Start with fewer results
browser search "topic" -n 3 --content
```

### 6. Use --profile for Authenticated Sites

```bash
# ✅ CORRECT - for sites requiring login
browser start --profile
browser nav https://gmail.com

# ❌ WRONG - will show login page
browser start
browser nav https://gmail.com
```

### 7. Validate Before Complex Operations

```bash
# Check if element exists before complex extraction
browser eval "document.querySelector('.data-table') !== null"

# If true, proceed with extraction
browser eval "/* complex extraction code */"
```

---

## Limitations

1. **No JavaScript execution on navigation**: Cannot interact with page elements (click, type, etc.). Only navigation and evaluation.

2. **Single browser instance**: Only one browser can run at a time. Starting a new one kills the existing instance.

3. **macOS only by default**: Chrome path is hardcoded for macOS. Needs modification for Windows/Linux.

4. **Viewport screenshots only**: Cannot capture full page, only visible area.

5. **No iframe access**: JavaScript evaluation doesn't automatically access iframe content.

6. **Google search only**: The search command only works with Google.

7. **Interactive picker requires human**: The `pick` command cannot be used in fully automated scenarios.

---

## Advanced JavaScript Evaluation Examples

### Extract all images
```bash
browser eval "Array.from(document.querySelectorAll('img')).map(img => ({src: img.src, alt: img.alt}))"
```

### Get all external links
```bash
browser eval "Array.from(document.querySelectorAll('a[href^=\"http\"]')).map(a => a.href)"
```

### Find broken images
```bash
browser eval "Array.from(document.querySelectorAll('img')).filter(img => !img.complete || img.naturalHeight === 0).map(img => img.src)"
```

### Get page performance metrics
```bash
browser eval "JSON.stringify(performance.timing)"
```

### Extract structured data (JSON-LD)
```bash
browser eval "Array.from(document.querySelectorAll('script[type=\"application/ld+json\"]')).map(s => JSON.parse(s.textContent))"
```

### Check for specific text on page
```bash
browser eval "document.body.textContent.includes('specific text')"
```

### Get all CSS classes used on page
```bash
browser eval "Array.from(new Set(Array.from(document.querySelectorAll('*')).flatMap(el => Array.from(el.classList))))"
```

### Extract OpenGraph metadata
```bash
browser eval "Object.fromEntries(Array.from(document.querySelectorAll('meta[property^=\"og:\"]')).map(m => [m.property, m.content]))"
```

---

## Complete Example: News Article Scraper

```bash
# 1. Start browser
browser start

# 2. Search for articles on a topic
browser search "artificial intelligence breakthroughs 2024" -n 5 > search_results.txt

# 3. Navigate to first result (extract URL from search_results.txt manually or programmatically)
browser nav https://example.com/ai-article

# 4. Extract article content
browser content > ai_article.md

# 5. Get article metadata
browser eval "({
  title: document.title,
  author: document.querySelector('meta[name=\"author\"]')?.content,
  published: document.querySelector('meta[property=\"article:published_time\"]')?.content,
  wordCount: document.body.textContent.trim().split(/\\s+/).length
})" > ai_article_meta.json

# 6. Take screenshot
browser screenshot

# 7. Check for related articles
browser eval "Array.from(document.querySelectorAll('.related-article')).map(a => ({
  title: a.querySelector('.title').textContent,
  url: a.href
}))" > related_articles.json
```

---

## Summary for AI Agents

**Key Points:**
1. **ALWAYS** start with `browser start`
2. Commands are **sequential** - wait for each to complete
3. Use **quotes** around JavaScript code and search queries
4. **Save output** to files when you need to preserve it
5. Use `browser eval` for **any** data extraction
6. The browser is **visible** - human can see what's happening
7. Only **one browser** instance at a time
8. Use `--profile` flag when you need **saved logins**

**Most Common Pattern:**
```bash
browser start
browser nav <url>
browser eval "<javascript>"
```

This covers 90% of use cases.
