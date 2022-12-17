# progress

**opinion mining**

- 是否只考虑右方位元素？
  - 左方位包含很多重要的修饰单位，例如`det`, `advmod`, `attr`。可能对empathic generation有较为重要的影响。应该尝试加入。
- 限制`go_deeper`的depth？
  - 很可能不需要，不过在加入左方位元素后，可能内容有所冗余。
- 验证`vp`的主语是wikipedia page 的`title`？
  - 有必要，有的`vp`不是在修饰页面的主题元素，应该过滤这些`vp`。