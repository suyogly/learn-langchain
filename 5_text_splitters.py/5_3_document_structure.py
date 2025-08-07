from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader

loader = '''

# Heading 1 (H1)
## Heading 2 (H2)
### Heading 3 (H3)
#### Heading 4 (H4)
##### Heading 5 (H5)
###### Heading 6 (H6)

---

This is a **bold** text.  
This is an *italic* text.  
This is ***bold and italic*** text.

> This is a blockquote.  
> Used to quote content or highlight information.

---

### Unordered List

- Item 1
  - Subitem 1.1
  - Subitem 1.2
- Item 2

### Ordered List

1. First item
2. Second item
3. Third item

---

### Code Examples

Inline code: `print("Hello, Markdown!")`

Multiline code block:

```python
def greet(name):
    return f"Hello, {name}!"
print(greet("World"))


'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap = 0
)

split = splitter.split_text(loader)

print(split[0])
print("---")
print(split[1])