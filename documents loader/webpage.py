from langchain_community.document_loaders import WebBaseLoader

url="https://www.apple.com/in/iphone-18-pro/?afid=p240%7Cgo~cmp-24228438609~adg-205746684651~ad-824402488477_kwd-308199750716~dev-c~ext-~prd-~mca-~nt-search&cid=wwa-in-kwgo-iphone-noncore_iphone18pro-iphone18pro-iphone18pro_hero_preorder_091226-iPhone18Pro-iPhone18Pro"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)