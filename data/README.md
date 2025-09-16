# 📊 E-commerce Dataset

## 📁 Dataset Files

### 🎯 **amazon_com_ecommerce.csv** (Primary Dataset)
- **Size**: 19MB
- **Products**: ~10,000 Amazon e-commerce products
- **Format**: CSV with comprehensive product information

#### 📋 **Columns:**
- `Uniq Id` - Unique product identifier
- `Product Name` - Full product name/title
- `Brand Name` - Product brand
- `Asin` - Amazon Standard Identification Number
- `Category` - Product category hierarchy
- `List Price` - Original price
- `Selling Price` - Current selling price
- `About Product` - Product description and features
- `Product Specification` - Technical specifications
- `Technical Details` - Detailed technical information
- `Shipping Weight` - Product weight
- `Product Dimensions` - Physical dimensions
- `Image` - Product image URLs
- `Product Url` - Amazon product page URL
- `Product Description` - Detailed description

### 🧪 **test_sample_1000.csv** (Test Dataset)
- **Size**: 1.9MB
- **Products**: 1,000 sample products
- **Purpose**: Testing and development
- **Format**: Subset of the main dataset

## 🚀 **Usage in RAG Pipeline**

### **For Production:**
```python
# Load full dataset (10K products)
df = pd.read_csv("data/amazon_com_ecommerce.csv")
```

### **For Testing:**
```python
# Load sample dataset (1K products)
df = pd.read_csv("data/test_sample_1000.csv")
```

### **For HuggingFace Spaces:**
```python
# Automatic detection in app.py
possible_paths = [
    "data/amazon_com_ecommerce.csv",  # Full dataset
    "data/test_sample_1000.csv",      # Test dataset
    "amazon_com_ecommerce.csv",       # If uploaded to Space root
    "products.csv"                    # Generic name
]
```

## 📊 **Dataset Statistics**

### **Full Dataset:**
- **Total Products**: ~10,000
- **Categories**: Electronics, Toys & Games, Sports & Outdoors, etc.
- **Price Range**: $0.99 - $2,000+
- **Complete Information**: Product names, descriptions, specifications, images

### **Data Quality:**
- ✅ Rich product descriptions
- ✅ Multiple image URLs
- ✅ Category hierarchies
- ✅ Technical specifications
- ✅ Pricing information
- ✅ Amazon URLs for verification

## 🎯 **Perfect for RAG Pipeline**

This dataset is ideal for the e-commerce RAG pipeline because it provides:

1. **Rich Text Content**: Detailed descriptions for embedding generation
2. **Multimodal Data**: Text + image URLs for CLIP embeddings
3. **Structured Information**: Categories, prices, specifications
4. **Real E-commerce Data**: Actual Amazon products with real URLs
5. **Diverse Categories**: Wide range of product types
6. **Complete Metadata**: All information needed for recommendations

## 🔧 **Integration Notes**

- **Column Mapping**: The `app_with_real_data.py` automatically maps various column name formats
- **Fallback Handling**: If dataset not found, falls back to mock data
- **Performance**: Loads first 100 products for demo performance
- **Scalability**: Can handle full 10K dataset in production

## 📈 **Recommended Usage**

### **Development/Testing:**
- Use `test_sample_1000.csv` for faster iteration
- Perfect for testing RAG pipeline components

### **Demo/Presentation:**
- Upload to HuggingFace Spaces for live demonstration
- Shows real Amazon products with working links

### **Production:**
- Use full `amazon_com_ecommerce.csv` dataset
- Implement proper vector indexing for 10K products
- Add caching for better performance

## 🎉 **Benefits**

This dataset transforms your RAG pipeline from a proof-of-concept to a **production-ready e-commerce search system** with real, comprehensive product data!