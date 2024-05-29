import{s as qn,x as Et,n as Lt,b as Kn,y as yn,z as kn,A as za}from"./scheduler.mhZtr3qf.js";import{S as Qn,i as Zn,E as Ra,H as Fa,F as ja,a as Da,C as Ya,d as Yt,G as Ua,g as Jn,m as xn,p as wt,r as Ha,n as Ut,o as Wa,I as Va,v as Ga,w as Xa,x as Ba,y as qa}from"./index.By0cP33f.js";function Ka(t,n){const a={},e={},r={$$scope:1};let i=t.length;for(;i--;){const o=t[i],s=n[i];if(s){for(const f in o)f in s||(e[f]=1);for(const f in s)r[f]||(a[f]=s[f],r[f]=1);t[i]=s}else for(const f in o)r[f]=1}for(const o in e)o in a||(a[o]=void 0);return a}function Qa(t){return typeof t=="object"&&t!==null?t:{}}function Za(t){const{beat:n,fade:a,beatFade:e,bounce:r,shake:i,flash:o,spin:s,spinPulse:f,spinReverse:c,pulse:l,fixedWidth:m,inverse:g,border:h,listItem:b,flip:p,size:w,rotation:E,pull:y}=t,S={"fa-beat":n,"fa-fade":a,"fa-beat-fade":e,"fa-bounce":r,"fa-shake":i,"fa-flash":o,"fa-spin":s,"fa-spin-reverse":c,"fa-spin-pulse":f,"fa-pulse":l,"fa-fw":m,"fa-inverse":g,"fa-border":h,"fa-li":b,"fa-flip":p===!0,"fa-flip-horizontal":p==="horizontal"||p==="both","fa-flip-vertical":p==="vertical"||p==="both",[`fa-${w}`]:typeof w<"u"&&w!==null,[`fa-rotate-${E}`]:typeof E<"u"&&E!==null&&E!==0,[`fa-pull-${y}`]:typeof y<"u"&&y!==null,"fa-swap-opacity":t.swapOpacity};return Object.keys(S).map(P=>S[P]?P:null).filter(P=>P)}function Ja(t){return t=t-0,t===t}function $a(t){return Ja(t)?t:(t=t.replace(/[\-_\s]+(.)?/g,function(n,a){return a?a.toUpperCase():""}),t.substr(0,1).toLowerCase()+t.substr(1))}function te(t){return typeof t=="string"?t:Object.keys(t).reduce((n,a)=>n+a.split(/(?=[A-Z])/).join("-").toLowerCase()+":"+t[a]+";","")}function $n(t,n,a={}){if(typeof n=="string")return n;const e=(n.children||[]).map(i=>$n(t,i)),r=Object.keys(n.attributes||{}).reduce((i,o)=>{const s=n.attributes[o];return o==="style"?i.attrs.style=te(s):o.indexOf("aria-")===0||o.indexOf("data-")===0?i.attrs[o.toLowerCase()]=s:i.attrs[$a(o)]=s,i},{attrs:{}});return t(n.tag,{...r.attrs},e)}function wn(t,n){var a=Object.keys(t);if(Object.getOwnPropertySymbols){var e=Object.getOwnPropertySymbols(t);n&&(e=e.filter(function(r){return Object.getOwnPropertyDescriptor(t,r).enumerable})),a.push.apply(a,e)}return a}function u(t){for(var n=1;n<arguments.length;n++){var a=arguments[n]!=null?arguments[n]:{};n%2?wn(Object(a),!0).forEach(function(e){O(t,e,a[e])}):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(a)):wn(Object(a)).forEach(function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(a,e))})}return t}function Ot(t){"@babel/helpers - typeof";return Ot=typeof Symbol=="function"&&typeof Symbol.iterator=="symbol"?function(n){return typeof n}:function(n){return n&&typeof Symbol=="function"&&n.constructor===Symbol&&n!==Symbol.prototype?"symbol":typeof n},Ot(t)}function ne(t,n){if(!(t instanceof n))throw new TypeError("Cannot call a class as a function")}function ae(t,n){for(var a=0;a<n.length;a++){var e=n[a];e.enumerable=e.enumerable||!1,e.configurable=!0,"value"in e&&(e.writable=!0),Object.defineProperty(t,e.key,e)}}function ee(t,n,a){return n&&ae(t.prototype,n),Object.defineProperty(t,"prototype",{writable:!1}),t}function O(t,n,a){return n in t?Object.defineProperty(t,n,{value:a,enumerable:!0,configurable:!0,writable:!0}):t[n]=a,t}function tn(t,n){return ie(t)||se(t,n)||ta(t,n)||le()}function lt(t){return re(t)||oe(t)||ta(t)||fe()}function re(t){if(Array.isArray(t))return Ht(t)}function ie(t){if(Array.isArray(t))return t}function oe(t){if(typeof Symbol<"u"&&t[Symbol.iterator]!=null||t["@@iterator"]!=null)return Array.from(t)}function se(t,n){var a=t==null?null:typeof Symbol<"u"&&t[Symbol.iterator]||t["@@iterator"];if(a!=null){var e=[],r=!0,i=!1,o,s;try{for(a=a.call(t);!(r=(o=a.next()).done)&&(e.push(o.value),!(n&&e.length===n));r=!0);}catch(f){i=!0,s=f}finally{try{!r&&a.return!=null&&a.return()}finally{if(i)throw s}}return e}}function ta(t,n){if(t){if(typeof t=="string")return Ht(t,n);var a=Object.prototype.toString.call(t).slice(8,-1);if(a==="Object"&&t.constructor&&(a=t.constructor.name),a==="Map"||a==="Set")return Array.from(t);if(a==="Arguments"||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(a))return Ht(t,n)}}function Ht(t,n){(n==null||n>t.length)&&(n=t.length);for(var a=0,e=new Array(n);a<n;a++)e[a]=t[a];return e}function fe(){throw new TypeError(`Invalid attempt to spread non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}function le(){throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`)}var An=function(){},nn={},na={},aa=null,ea={mark:An,measure:An};try{typeof window<"u"&&(nn=window),typeof document<"u"&&(na=document),typeof MutationObserver<"u"&&(aa=MutationObserver),typeof performance<"u"&&(ea=performance)}catch{}var ce=nn.navigator||{},Sn=ce.userAgent,En=Sn===void 0?"":Sn,Y=nn,x=na,On=aa,dt=ea;Y.document;var j=!!x.documentElement&&!!x.head&&typeof x.addEventListener=="function"&&typeof x.createElement=="function",ra=~En.indexOf("MSIE")||~En.indexOf("Trident/"),vt,gt,bt,ht,pt,z="___FONT_AWESOME___",Wt=16,ia="fa",oa="svg-inline--fa",G="data-fa-i2svg",Vt="data-fa-pseudo-element",ue="data-fa-pseudo-element-pending",an="data-prefix",en="data-icon",Pn="fontawesome-i2svg",me="async",de=["HTML","HEAD","STYLE","SCRIPT"],sa=function(){try{return!0}catch{return!1}}(),k="classic",A="sharp",rn=[k,A];function ct(t){return new Proxy(t,{get:function(a,e){return e in a?a[e]:a[k]}})}var rt=ct((vt={},O(vt,k,{fa:"solid",fas:"solid","fa-solid":"solid",far:"regular","fa-regular":"regular",fal:"light","fa-light":"light",fat:"thin","fa-thin":"thin",fad:"duotone","fa-duotone":"duotone",fab:"brands","fa-brands":"brands",fak:"kit",fakd:"kit","fa-kit":"kit","fa-kit-duotone":"kit"}),O(vt,A,{fa:"solid",fass:"solid","fa-solid":"solid",fasr:"regular","fa-regular":"regular",fasl:"light","fa-light":"light",fast:"thin","fa-thin":"thin"}),vt)),it=ct((gt={},O(gt,k,{solid:"fas",regular:"far",light:"fal",thin:"fat",duotone:"fad",brands:"fab",kit:"fak"}),O(gt,A,{solid:"fass",regular:"fasr",light:"fasl",thin:"fast"}),gt)),ot=ct((bt={},O(bt,k,{fab:"fa-brands",fad:"fa-duotone",fak:"fa-kit",fal:"fa-light",far:"fa-regular",fas:"fa-solid",fat:"fa-thin"}),O(bt,A,{fass:"fa-solid",fasr:"fa-regular",fasl:"fa-light",fast:"fa-thin"}),bt)),ve=ct((ht={},O(ht,k,{"fa-brands":"fab","fa-duotone":"fad","fa-kit":"fak","fa-light":"fal","fa-regular":"far","fa-solid":"fas","fa-thin":"fat"}),O(ht,A,{"fa-solid":"fass","fa-regular":"fasr","fa-light":"fasl","fa-thin":"fast"}),ht)),ge=/fa(s|r|l|t|d|b|k|ss|sr|sl|st)?[\-\ ]/,fa="fa-layers-text",be=/Font ?Awesome ?([56 ]*)(Solid|Regular|Light|Thin|Duotone|Brands|Free|Pro|Sharp|Kit)?.*/i,he=ct((pt={},O(pt,k,{900:"fas",400:"far",normal:"far",300:"fal",100:"fat"}),O(pt,A,{900:"fass",400:"fasr",300:"fasl",100:"fast"}),pt)),la=[1,2,3,4,5,6,7,8,9,10],pe=la.concat([11,12,13,14,15,16,17,18,19,20]),ye=["class","data-prefix","data-icon","data-fa-transform","data-fa-mask"],W={GROUP:"duotone-group",SWAP_OPACITY:"swap-opacity",PRIMARY:"primary",SECONDARY:"secondary"},st=new Set;Object.keys(it[k]).map(st.add.bind(st));Object.keys(it[A]).map(st.add.bind(st));var ke=[].concat(rn,lt(st),["2xs","xs","sm","lg","xl","2xl","beat","border","fade","beat-fade","bounce","flip-both","flip-horizontal","flip-vertical","flip","fw","inverse","layers-counter","layers-text","layers","li","pull-left","pull-right","pulse","rotate-180","rotate-270","rotate-90","rotate-by","shake","spin-pulse","spin-reverse","spin","stack-1x","stack-2x","stack","ul",W.GROUP,W.SWAP_OPACITY,W.PRIMARY,W.SECONDARY]).concat(la.map(function(t){return"".concat(t,"x")})).concat(pe.map(function(t){return"w-".concat(t)})),at=Y.FontAwesomeConfig||{};function xe(t){var n=x.querySelector("script["+t+"]");if(n)return n.getAttribute(t)}function we(t){return t===""?!0:t==="false"?!1:t==="true"?!0:t}if(x&&typeof x.querySelector=="function"){var Ae=[["data-family-prefix","familyPrefix"],["data-css-prefix","cssPrefix"],["data-family-default","familyDefault"],["data-style-default","styleDefault"],["data-replacement-class","replacementClass"],["data-auto-replace-svg","autoReplaceSvg"],["data-auto-add-css","autoAddCss"],["data-auto-a11y","autoA11y"],["data-search-pseudo-elements","searchPseudoElements"],["data-observe-mutations","observeMutations"],["data-mutate-approach","mutateApproach"],["data-keep-original-source","keepOriginalSource"],["data-measure-performance","measurePerformance"],["data-show-missing-icons","showMissingIcons"]];Ae.forEach(function(t){var n=tn(t,2),a=n[0],e=n[1],r=we(xe(a));r!=null&&(at[e]=r)})}var ca={styleDefault:"solid",familyDefault:"classic",cssPrefix:ia,replacementClass:oa,autoReplaceSvg:!0,autoAddCss:!0,autoA11y:!0,searchPseudoElements:!1,observeMutations:!0,mutateApproach:"async",keepOriginalSource:!0,measurePerformance:!1,showMissingIcons:!0};at.familyPrefix&&(at.cssPrefix=at.familyPrefix);var Z=u(u({},ca),at);Z.autoReplaceSvg||(Z.observeMutations=!1);var d={};Object.keys(ca).forEach(function(t){Object.defineProperty(d,t,{enumerable:!0,set:function(a){Z[t]=a,et.forEach(function(e){return e(d)})},get:function(){return Z[t]}})});Object.defineProperty(d,"familyPrefix",{enumerable:!0,set:function(n){Z.cssPrefix=n,et.forEach(function(a){return a(d)})},get:function(){return Z.cssPrefix}});Y.FontAwesomeConfig=d;var et=[];function Se(t){return et.push(t),function(){et.splice(et.indexOf(t),1)}}var D=Wt,L={size:16,x:0,y:0,rotate:0,flipX:!1,flipY:!1};function Ee(t){if(!(!t||!j)){var n=x.createElement("style");n.setAttribute("type","text/css"),n.innerHTML=t;for(var a=x.head.childNodes,e=null,r=a.length-1;r>-1;r--){var i=a[r],o=(i.tagName||"").toUpperCase();["STYLE","LINK"].indexOf(o)>-1&&(e=i)}return x.head.insertBefore(n,e),t}}var Oe="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";function ft(){for(var t=12,n="";t-- >0;)n+=Oe[Math.random()*62|0];return n}function J(t){for(var n=[],a=(t||[]).length>>>0;a--;)n[a]=t[a];return n}function on(t){return t.classList?J(t.classList):(t.getAttribute("class")||"").split(" ").filter(function(n){return n})}function ua(t){return"".concat(t).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/'/g,"&#39;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function Pe(t){return Object.keys(t||{}).reduce(function(n,a){return n+"".concat(a,'="').concat(ua(t[a]),'" ')},"").trim()}function It(t){return Object.keys(t||{}).reduce(function(n,a){return n+"".concat(a,": ").concat(t[a].trim(),";")},"")}function sn(t){return t.size!==L.size||t.x!==L.x||t.y!==L.y||t.rotate!==L.rotate||t.flipX||t.flipY}function Ce(t){var n=t.transform,a=t.containerWidth,e=t.iconWidth,r={transform:"translate(".concat(a/2," 256)")},i="translate(".concat(n.x*32,", ").concat(n.y*32,") "),o="scale(".concat(n.size/16*(n.flipX?-1:1),", ").concat(n.size/16*(n.flipY?-1:1),") "),s="rotate(".concat(n.rotate," 0 0)"),f={transform:"".concat(i," ").concat(o," ").concat(s)},c={transform:"translate(".concat(e/2*-1," -256)")};return{outer:r,inner:f,path:c}}function Ie(t){var n=t.transform,a=t.width,e=a===void 0?Wt:a,r=t.height,i=r===void 0?Wt:r,o=t.startCentered,s=o===void 0?!1:o,f="";return s&&ra?f+="translate(".concat(n.x/D-e/2,"em, ").concat(n.y/D-i/2,"em) "):s?f+="translate(calc(-50% + ".concat(n.x/D,"em), calc(-50% + ").concat(n.y/D,"em)) "):f+="translate(".concat(n.x/D,"em, ").concat(n.y/D,"em) "),f+="scale(".concat(n.size/D*(n.flipX?-1:1),", ").concat(n.size/D*(n.flipY?-1:1),") "),f+="rotate(".concat(n.rotate,"deg) "),f}var Ne=`:root, :host {
  --fa-font-solid: normal 900 1em/1 "Font Awesome 6 Solid";
  --fa-font-regular: normal 400 1em/1 "Font Awesome 6 Regular";
  --fa-font-light: normal 300 1em/1 "Font Awesome 6 Light";
  --fa-font-thin: normal 100 1em/1 "Font Awesome 6 Thin";
  --fa-font-duotone: normal 900 1em/1 "Font Awesome 6 Duotone";
  --fa-font-sharp-solid: normal 900 1em/1 "Font Awesome 6 Sharp";
  --fa-font-sharp-regular: normal 400 1em/1 "Font Awesome 6 Sharp";
  --fa-font-sharp-light: normal 300 1em/1 "Font Awesome 6 Sharp";
  --fa-font-sharp-thin: normal 100 1em/1 "Font Awesome 6 Sharp";
  --fa-font-brands: normal 400 1em/1 "Font Awesome 6 Brands";
}

svg:not(:root).svg-inline--fa, svg:not(:host).svg-inline--fa {
  overflow: visible;
  box-sizing: content-box;
}

.svg-inline--fa {
  display: var(--fa-display, inline-block);
  height: 1em;
  overflow: visible;
  vertical-align: -0.125em;
}
.svg-inline--fa.fa-2xs {
  vertical-align: 0.1em;
}
.svg-inline--fa.fa-xs {
  vertical-align: 0em;
}
.svg-inline--fa.fa-sm {
  vertical-align: -0.0714285705em;
}
.svg-inline--fa.fa-lg {
  vertical-align: -0.2em;
}
.svg-inline--fa.fa-xl {
  vertical-align: -0.25em;
}
.svg-inline--fa.fa-2xl {
  vertical-align: -0.3125em;
}
.svg-inline--fa.fa-pull-left {
  margin-right: var(--fa-pull-margin, 0.3em);
  width: auto;
}
.svg-inline--fa.fa-pull-right {
  margin-left: var(--fa-pull-margin, 0.3em);
  width: auto;
}
.svg-inline--fa.fa-li {
  width: var(--fa-li-width, 2em);
  top: 0.25em;
}
.svg-inline--fa.fa-fw {
  width: var(--fa-fw-width, 1.25em);
}

.fa-layers svg.svg-inline--fa {
  bottom: 0;
  left: 0;
  margin: auto;
  position: absolute;
  right: 0;
  top: 0;
}

.fa-layers-counter, .fa-layers-text {
  display: inline-block;
  position: absolute;
  text-align: center;
}

.fa-layers {
  display: inline-block;
  height: 1em;
  position: relative;
  text-align: center;
  vertical-align: -0.125em;
  width: 1em;
}
.fa-layers svg.svg-inline--fa {
  -webkit-transform-origin: center center;
          transform-origin: center center;
}

.fa-layers-text {
  left: 50%;
  top: 50%;
  -webkit-transform: translate(-50%, -50%);
          transform: translate(-50%, -50%);
  -webkit-transform-origin: center center;
          transform-origin: center center;
}

.fa-layers-counter {
  background-color: var(--fa-counter-background-color, #ff253a);
  border-radius: var(--fa-counter-border-radius, 1em);
  box-sizing: border-box;
  color: var(--fa-inverse, #fff);
  line-height: var(--fa-counter-line-height, 1);
  max-width: var(--fa-counter-max-width, 5em);
  min-width: var(--fa-counter-min-width, 1.5em);
  overflow: hidden;
  padding: var(--fa-counter-padding, 0.25em 0.5em);
  right: var(--fa-right, 0);
  text-overflow: ellipsis;
  top: var(--fa-top, 0);
  -webkit-transform: scale(var(--fa-counter-scale, 0.25));
          transform: scale(var(--fa-counter-scale, 0.25));
  -webkit-transform-origin: top right;
          transform-origin: top right;
}

.fa-layers-bottom-right {
  bottom: var(--fa-bottom, 0);
  right: var(--fa-right, 0);
  top: auto;
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: bottom right;
          transform-origin: bottom right;
}

.fa-layers-bottom-left {
  bottom: var(--fa-bottom, 0);
  left: var(--fa-left, 0);
  right: auto;
  top: auto;
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: bottom left;
          transform-origin: bottom left;
}

.fa-layers-top-right {
  top: var(--fa-top, 0);
  right: var(--fa-right, 0);
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: top right;
          transform-origin: top right;
}

.fa-layers-top-left {
  left: var(--fa-left, 0);
  right: auto;
  top: var(--fa-top, 0);
  -webkit-transform: scale(var(--fa-layers-scale, 0.25));
          transform: scale(var(--fa-layers-scale, 0.25));
  -webkit-transform-origin: top left;
          transform-origin: top left;
}

.fa-1x {
  font-size: 1em;
}

.fa-2x {
  font-size: 2em;
}

.fa-3x {
  font-size: 3em;
}

.fa-4x {
  font-size: 4em;
}

.fa-5x {
  font-size: 5em;
}

.fa-6x {
  font-size: 6em;
}

.fa-7x {
  font-size: 7em;
}

.fa-8x {
  font-size: 8em;
}

.fa-9x {
  font-size: 9em;
}

.fa-10x {
  font-size: 10em;
}

.fa-2xs {
  font-size: 0.625em;
  line-height: 0.1em;
  vertical-align: 0.225em;
}

.fa-xs {
  font-size: 0.75em;
  line-height: 0.0833333337em;
  vertical-align: 0.125em;
}

.fa-sm {
  font-size: 0.875em;
  line-height: 0.0714285718em;
  vertical-align: 0.0535714295em;
}

.fa-lg {
  font-size: 1.25em;
  line-height: 0.05em;
  vertical-align: -0.075em;
}

.fa-xl {
  font-size: 1.5em;
  line-height: 0.0416666682em;
  vertical-align: -0.125em;
}

.fa-2xl {
  font-size: 2em;
  line-height: 0.03125em;
  vertical-align: -0.1875em;
}

.fa-fw {
  text-align: center;
  width: 1.25em;
}

.fa-ul {
  list-style-type: none;
  margin-left: var(--fa-li-margin, 2.5em);
  padding-left: 0;
}
.fa-ul > li {
  position: relative;
}

.fa-li {
  left: calc(var(--fa-li-width, 2em) * -1);
  position: absolute;
  text-align: center;
  width: var(--fa-li-width, 2em);
  line-height: inherit;
}

.fa-border {
  border-color: var(--fa-border-color, #eee);
  border-radius: var(--fa-border-radius, 0.1em);
  border-style: var(--fa-border-style, solid);
  border-width: var(--fa-border-width, 0.08em);
  padding: var(--fa-border-padding, 0.2em 0.25em 0.15em);
}

.fa-pull-left {
  float: left;
  margin-right: var(--fa-pull-margin, 0.3em);
}

.fa-pull-right {
  float: right;
  margin-left: var(--fa-pull-margin, 0.3em);
}

.fa-beat {
  -webkit-animation-name: fa-beat;
          animation-name: fa-beat;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, ease-in-out);
          animation-timing-function: var(--fa-animation-timing, ease-in-out);
}

.fa-bounce {
  -webkit-animation-name: fa-bounce;
          animation-name: fa-bounce;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.28, 0.84, 0.42, 1));
          animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.28, 0.84, 0.42, 1));
}

.fa-fade {
  -webkit-animation-name: fa-fade;
          animation-name: fa-fade;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
          animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
}

.fa-beat-fade {
  -webkit-animation-name: fa-beat-fade;
          animation-name: fa-beat-fade;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
          animation-timing-function: var(--fa-animation-timing, cubic-bezier(0.4, 0, 0.6, 1));
}

.fa-flip {
  -webkit-animation-name: fa-flip;
          animation-name: fa-flip;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, ease-in-out);
          animation-timing-function: var(--fa-animation-timing, ease-in-out);
}

.fa-shake {
  -webkit-animation-name: fa-shake;
          animation-name: fa-shake;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, linear);
          animation-timing-function: var(--fa-animation-timing, linear);
}

.fa-spin {
  -webkit-animation-name: fa-spin;
          animation-name: fa-spin;
  -webkit-animation-delay: var(--fa-animation-delay, 0s);
          animation-delay: var(--fa-animation-delay, 0s);
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 2s);
          animation-duration: var(--fa-animation-duration, 2s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, linear);
          animation-timing-function: var(--fa-animation-timing, linear);
}

.fa-spin-reverse {
  --fa-animation-direction: reverse;
}

.fa-pulse,
.fa-spin-pulse {
  -webkit-animation-name: fa-spin;
          animation-name: fa-spin;
  -webkit-animation-direction: var(--fa-animation-direction, normal);
          animation-direction: var(--fa-animation-direction, normal);
  -webkit-animation-duration: var(--fa-animation-duration, 1s);
          animation-duration: var(--fa-animation-duration, 1s);
  -webkit-animation-iteration-count: var(--fa-animation-iteration-count, infinite);
          animation-iteration-count: var(--fa-animation-iteration-count, infinite);
  -webkit-animation-timing-function: var(--fa-animation-timing, steps(8));
          animation-timing-function: var(--fa-animation-timing, steps(8));
}

@media (prefers-reduced-motion: reduce) {
  .fa-beat,
.fa-bounce,
.fa-fade,
.fa-beat-fade,
.fa-flip,
.fa-pulse,
.fa-shake,
.fa-spin,
.fa-spin-pulse {
    -webkit-animation-delay: -1ms;
            animation-delay: -1ms;
    -webkit-animation-duration: 1ms;
            animation-duration: 1ms;
    -webkit-animation-iteration-count: 1;
            animation-iteration-count: 1;
    -webkit-transition-delay: 0s;
            transition-delay: 0s;
    -webkit-transition-duration: 0s;
            transition-duration: 0s;
  }
}
@-webkit-keyframes fa-beat {
  0%, 90% {
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  45% {
    -webkit-transform: scale(var(--fa-beat-scale, 1.25));
            transform: scale(var(--fa-beat-scale, 1.25));
  }
}
@keyframes fa-beat {
  0%, 90% {
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  45% {
    -webkit-transform: scale(var(--fa-beat-scale, 1.25));
            transform: scale(var(--fa-beat-scale, 1.25));
  }
}
@-webkit-keyframes fa-bounce {
  0% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  10% {
    -webkit-transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
            transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
  }
  30% {
    -webkit-transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
            transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
  }
  50% {
    -webkit-transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
            transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
  }
  57% {
    -webkit-transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
            transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
  }
  64% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  100% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
}
@keyframes fa-bounce {
  0% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  10% {
    -webkit-transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
            transform: scale(var(--fa-bounce-start-scale-x, 1.1), var(--fa-bounce-start-scale-y, 0.9)) translateY(0);
  }
  30% {
    -webkit-transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
            transform: scale(var(--fa-bounce-jump-scale-x, 0.9), var(--fa-bounce-jump-scale-y, 1.1)) translateY(var(--fa-bounce-height, -0.5em));
  }
  50% {
    -webkit-transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
            transform: scale(var(--fa-bounce-land-scale-x, 1.05), var(--fa-bounce-land-scale-y, 0.95)) translateY(0);
  }
  57% {
    -webkit-transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
            transform: scale(1, 1) translateY(var(--fa-bounce-rebound, -0.125em));
  }
  64% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
  100% {
    -webkit-transform: scale(1, 1) translateY(0);
            transform: scale(1, 1) translateY(0);
  }
}
@-webkit-keyframes fa-fade {
  50% {
    opacity: var(--fa-fade-opacity, 0.4);
  }
}
@keyframes fa-fade {
  50% {
    opacity: var(--fa-fade-opacity, 0.4);
  }
}
@-webkit-keyframes fa-beat-fade {
  0%, 100% {
    opacity: var(--fa-beat-fade-opacity, 0.4);
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  50% {
    opacity: 1;
    -webkit-transform: scale(var(--fa-beat-fade-scale, 1.125));
            transform: scale(var(--fa-beat-fade-scale, 1.125));
  }
}
@keyframes fa-beat-fade {
  0%, 100% {
    opacity: var(--fa-beat-fade-opacity, 0.4);
    -webkit-transform: scale(1);
            transform: scale(1);
  }
  50% {
    opacity: 1;
    -webkit-transform: scale(var(--fa-beat-fade-scale, 1.125));
            transform: scale(var(--fa-beat-fade-scale, 1.125));
  }
}
@-webkit-keyframes fa-flip {
  50% {
    -webkit-transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
            transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
  }
}
@keyframes fa-flip {
  50% {
    -webkit-transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
            transform: rotate3d(var(--fa-flip-x, 0), var(--fa-flip-y, 1), var(--fa-flip-z, 0), var(--fa-flip-angle, -180deg));
  }
}
@-webkit-keyframes fa-shake {
  0% {
    -webkit-transform: rotate(-15deg);
            transform: rotate(-15deg);
  }
  4% {
    -webkit-transform: rotate(15deg);
            transform: rotate(15deg);
  }
  8%, 24% {
    -webkit-transform: rotate(-18deg);
            transform: rotate(-18deg);
  }
  12%, 28% {
    -webkit-transform: rotate(18deg);
            transform: rotate(18deg);
  }
  16% {
    -webkit-transform: rotate(-22deg);
            transform: rotate(-22deg);
  }
  20% {
    -webkit-transform: rotate(22deg);
            transform: rotate(22deg);
  }
  32% {
    -webkit-transform: rotate(-12deg);
            transform: rotate(-12deg);
  }
  36% {
    -webkit-transform: rotate(12deg);
            transform: rotate(12deg);
  }
  40%, 100% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
}
@keyframes fa-shake {
  0% {
    -webkit-transform: rotate(-15deg);
            transform: rotate(-15deg);
  }
  4% {
    -webkit-transform: rotate(15deg);
            transform: rotate(15deg);
  }
  8%, 24% {
    -webkit-transform: rotate(-18deg);
            transform: rotate(-18deg);
  }
  12%, 28% {
    -webkit-transform: rotate(18deg);
            transform: rotate(18deg);
  }
  16% {
    -webkit-transform: rotate(-22deg);
            transform: rotate(-22deg);
  }
  20% {
    -webkit-transform: rotate(22deg);
            transform: rotate(22deg);
  }
  32% {
    -webkit-transform: rotate(-12deg);
            transform: rotate(-12deg);
  }
  36% {
    -webkit-transform: rotate(12deg);
            transform: rotate(12deg);
  }
  40%, 100% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(360deg);
            transform: rotate(360deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
            transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(360deg);
            transform: rotate(360deg);
  }
}
.fa-rotate-90 {
  -webkit-transform: rotate(90deg);
          transform: rotate(90deg);
}

.fa-rotate-180 {
  -webkit-transform: rotate(180deg);
          transform: rotate(180deg);
}

.fa-rotate-270 {
  -webkit-transform: rotate(270deg);
          transform: rotate(270deg);
}

.fa-flip-horizontal {
  -webkit-transform: scale(-1, 1);
          transform: scale(-1, 1);
}

.fa-flip-vertical {
  -webkit-transform: scale(1, -1);
          transform: scale(1, -1);
}

.fa-flip-both,
.fa-flip-horizontal.fa-flip-vertical {
  -webkit-transform: scale(-1, -1);
          transform: scale(-1, -1);
}

.fa-rotate-by {
  -webkit-transform: rotate(var(--fa-rotate-angle, 0));
          transform: rotate(var(--fa-rotate-angle, 0));
}

.fa-stack {
  display: inline-block;
  vertical-align: middle;
  height: 2em;
  position: relative;
  width: 2.5em;
}

.fa-stack-1x,
.fa-stack-2x {
  bottom: 0;
  left: 0;
  margin: auto;
  position: absolute;
  right: 0;
  top: 0;
  z-index: var(--fa-stack-z-index, auto);
}

.svg-inline--fa.fa-stack-1x {
  height: 1em;
  width: 1.25em;
}
.svg-inline--fa.fa-stack-2x {
  height: 2em;
  width: 2.5em;
}

.fa-inverse {
  color: var(--fa-inverse, #fff);
}

.sr-only,
.fa-sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.sr-only-focusable:not(:focus),
.fa-sr-only-focusable:not(:focus) {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.svg-inline--fa .fa-primary {
  fill: var(--fa-primary-color, currentColor);
  opacity: var(--fa-primary-opacity, 1);
}

.svg-inline--fa .fa-secondary {
  fill: var(--fa-secondary-color, currentColor);
  opacity: var(--fa-secondary-opacity, 0.4);
}

.svg-inline--fa.fa-swap-opacity .fa-primary {
  opacity: var(--fa-secondary-opacity, 0.4);
}

.svg-inline--fa.fa-swap-opacity .fa-secondary {
  opacity: var(--fa-primary-opacity, 1);
}

.svg-inline--fa mask .fa-primary,
.svg-inline--fa mask .fa-secondary {
  fill: black;
}

.fad.fa-inverse,
.fa-duotone.fa-inverse {
  color: var(--fa-inverse, #fff);
}`;function ma(){var t=ia,n=oa,a=d.cssPrefix,e=d.replacementClass,r=Ne;if(a!==t||e!==n){var i=new RegExp("\\.".concat(t,"\\-"),"g"),o=new RegExp("\\--".concat(t,"\\-"),"g"),s=new RegExp("\\.".concat(n),"g");r=r.replace(i,".".concat(a,"-")).replace(o,"--".concat(a,"-")).replace(s,".".concat(e))}return r}var Cn=!1;function zt(){d.autoAddCss&&!Cn&&(Ee(ma()),Cn=!0)}var Te={mixout:function(){return{dom:{css:ma,insertCss:zt}}},hooks:function(){return{beforeDOMElementCreation:function(){zt()},beforeI2svg:function(){zt()}}}},R=Y||{};R[z]||(R[z]={});R[z].styles||(R[z].styles={});R[z].hooks||(R[z].hooks={});R[z].shims||(R[z].shims=[]);var T=R[z],da=[],_e=function t(){x.removeEventListener("DOMContentLoaded",t),Pt=1,da.map(function(n){return n()})},Pt=!1;j&&(Pt=(x.documentElement.doScroll?/^loaded|^c/:/^loaded|^i|^c/).test(x.readyState),Pt||x.addEventListener("DOMContentLoaded",_e));function Me(t){j&&(Pt?setTimeout(t,0):da.push(t))}function ut(t){var n=t.tag,a=t.attributes,e=a===void 0?{}:a,r=t.children,i=r===void 0?[]:r;return typeof t=="string"?ua(t):"<".concat(n," ").concat(Pe(e),">").concat(i.map(ut).join(""),"</").concat(n,">")}function In(t,n,a){if(t&&t[n]&&t[n][a])return{prefix:n,iconName:a,icon:t[n][a]}}var Rt=function(n,a,e,r){var i=Object.keys(n),o=i.length,s=a,f,c,l;for(e===void 0?(f=1,l=n[i[0]]):(f=0,l=e);f<o;f++)c=i[f],l=s(l,n[c],c,n);return l};function Le(t){for(var n=[],a=0,e=t.length;a<e;){var r=t.charCodeAt(a++);if(r>=55296&&r<=56319&&a<e){var i=t.charCodeAt(a++);(i&64512)==56320?n.push(((r&1023)<<10)+(i&1023)+65536):(n.push(r),a--)}else n.push(r)}return n}function Gt(t){var n=Le(t);return n.length===1?n[0].toString(16):null}function ze(t,n){var a=t.length,e=t.charCodeAt(n),r;return e>=55296&&e<=56319&&a>n+1&&(r=t.charCodeAt(n+1),r>=56320&&r<=57343)?(e-55296)*1024+r-56320+65536:e}function Nn(t){return Object.keys(t).reduce(function(n,a){var e=t[a],r=!!e.icon;return r?n[e.iconName]=e.icon:n[a]=e,n},{})}function Xt(t,n){var a=arguments.length>2&&arguments[2]!==void 0?arguments[2]:{},e=a.skipHooks,r=e===void 0?!1:e,i=Nn(n);typeof T.hooks.addPack=="function"&&!r?T.hooks.addPack(t,Nn(n)):T.styles[t]=u(u({},T.styles[t]||{}),i),t==="fas"&&Xt("fa",n)}var yt,kt,xt,q=T.styles,Re=T.shims,Fe=(yt={},O(yt,k,Object.values(ot[k])),O(yt,A,Object.values(ot[A])),yt),fn=null,va={},ga={},ba={},ha={},pa={},je=(kt={},O(kt,k,Object.keys(rt[k])),O(kt,A,Object.keys(rt[A])),kt);function De(t){return~ke.indexOf(t)}function Ye(t,n){var a=n.split("-"),e=a[0],r=a.slice(1).join("-");return e===t&&r!==""&&!De(r)?r:null}var ya=function(){var n=function(i){return Rt(q,function(o,s,f){return o[f]=Rt(s,i,{}),o},{})};va=n(function(r,i,o){if(i[3]&&(r[i[3]]=o),i[2]){var s=i[2].filter(function(f){return typeof f=="number"});s.forEach(function(f){r[f.toString(16)]=o})}return r}),ga=n(function(r,i,o){if(r[o]=o,i[2]){var s=i[2].filter(function(f){return typeof f=="string"});s.forEach(function(f){r[f]=o})}return r}),pa=n(function(r,i,o){var s=i[2];return r[o]=o,s.forEach(function(f){r[f]=o}),r});var a="far"in q||d.autoFetchSvg,e=Rt(Re,function(r,i){var o=i[0],s=i[1],f=i[2];return s==="far"&&!a&&(s="fas"),typeof o=="string"&&(r.names[o]={prefix:s,iconName:f}),typeof o=="number"&&(r.unicodes[o.toString(16)]={prefix:s,iconName:f}),r},{names:{},unicodes:{}});ba=e.names,ha=e.unicodes,fn=Nt(d.styleDefault,{family:d.familyDefault})};Se(function(t){fn=Nt(t.styleDefault,{family:d.familyDefault})});ya();function ln(t,n){return(va[t]||{})[n]}function Ue(t,n){return(ga[t]||{})[n]}function V(t,n){return(pa[t]||{})[n]}function ka(t){return ba[t]||{prefix:null,iconName:null}}function He(t){var n=ha[t],a=ln("fas",t);return n||(a?{prefix:"fas",iconName:a}:null)||{prefix:null,iconName:null}}function U(){return fn}var cn=function(){return{prefix:null,iconName:null,rest:[]}};function Nt(t){var n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},a=n.family,e=a===void 0?k:a,r=rt[e][t],i=it[e][t]||it[e][r],o=t in T.styles?t:null;return i||o||null}var Tn=(xt={},O(xt,k,Object.keys(ot[k])),O(xt,A,Object.keys(ot[A])),xt);function Tt(t){var n,a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},e=a.skipLookups,r=e===void 0?!1:e,i=(n={},O(n,k,"".concat(d.cssPrefix,"-").concat(k)),O(n,A,"".concat(d.cssPrefix,"-").concat(A)),n),o=null,s=k;(t.includes(i[k])||t.some(function(c){return Tn[k].includes(c)}))&&(s=k),(t.includes(i[A])||t.some(function(c){return Tn[A].includes(c)}))&&(s=A);var f=t.reduce(function(c,l){var m=Ye(d.cssPrefix,l);if(q[l]?(l=Fe[s].includes(l)?ve[s][l]:l,o=l,c.prefix=l):je[s].indexOf(l)>-1?(o=l,c.prefix=Nt(l,{family:s})):m?c.iconName=m:l!==d.replacementClass&&l!==i[k]&&l!==i[A]&&c.rest.push(l),!r&&c.prefix&&c.iconName){var g=o==="fa"?ka(c.iconName):{},h=V(c.prefix,c.iconName);g.prefix&&(o=null),c.iconName=g.iconName||h||c.iconName,c.prefix=g.prefix||c.prefix,c.prefix==="far"&&!q.far&&q.fas&&!d.autoFetchSvg&&(c.prefix="fas")}return c},cn());return(t.includes("fa-brands")||t.includes("fab"))&&(f.prefix="fab"),(t.includes("fa-duotone")||t.includes("fad"))&&(f.prefix="fad"),!f.prefix&&s===A&&(q.fass||d.autoFetchSvg)&&(f.prefix="fass",f.iconName=V(f.prefix,f.iconName)||f.iconName),(f.prefix==="fa"||o==="fa")&&(f.prefix=U()||"fas"),f}var We=function(){function t(){ne(this,t),this.definitions={}}return ee(t,[{key:"add",value:function(){for(var a=this,e=arguments.length,r=new Array(e),i=0;i<e;i++)r[i]=arguments[i];var o=r.reduce(this._pullDefinitions,{});Object.keys(o).forEach(function(s){a.definitions[s]=u(u({},a.definitions[s]||{}),o[s]),Xt(s,o[s]);var f=ot[k][s];f&&Xt(f,o[s]),ya()})}},{key:"reset",value:function(){this.definitions={}}},{key:"_pullDefinitions",value:function(a,e){var r=e.prefix&&e.iconName&&e.icon?{0:e}:e;return Object.keys(r).map(function(i){var o=r[i],s=o.prefix,f=o.iconName,c=o.icon,l=c[2];a[s]||(a[s]={}),l.length>0&&l.forEach(function(m){typeof m=="string"&&(a[s][m]=c)}),a[s][f]=c}),a}}]),t}(),_n=[],K={},Q={},Ve=Object.keys(Q);function Ge(t,n){var a=n.mixoutsTo;return _n=t,K={},Object.keys(Q).forEach(function(e){Ve.indexOf(e)===-1&&delete Q[e]}),_n.forEach(function(e){var r=e.mixout?e.mixout():{};if(Object.keys(r).forEach(function(o){typeof r[o]=="function"&&(a[o]=r[o]),Ot(r[o])==="object"&&Object.keys(r[o]).forEach(function(s){a[o]||(a[o]={}),a[o][s]=r[o][s]})}),e.hooks){var i=e.hooks();Object.keys(i).forEach(function(o){K[o]||(K[o]=[]),K[o].push(i[o])})}e.provides&&e.provides(Q)}),a}function Bt(t,n){for(var a=arguments.length,e=new Array(a>2?a-2:0),r=2;r<a;r++)e[r-2]=arguments[r];var i=K[t]||[];return i.forEach(function(o){n=o.apply(null,[n].concat(e))}),n}function X(t){for(var n=arguments.length,a=new Array(n>1?n-1:0),e=1;e<n;e++)a[e-1]=arguments[e];var r=K[t]||[];r.forEach(function(i){i.apply(null,a)})}function F(){var t=arguments[0],n=Array.prototype.slice.call(arguments,1);return Q[t]?Q[t].apply(null,n):void 0}function qt(t){t.prefix==="fa"&&(t.prefix="fas");var n=t.iconName,a=t.prefix||U();if(n)return n=V(a,n)||n,In(xa.definitions,a,n)||In(T.styles,a,n)}var xa=new We,Xe=function(){d.autoReplaceSvg=!1,d.observeMutations=!1,X("noAuto")},Be={i2svg:function(){var n=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{};return j?(X("beforeI2svg",n),F("pseudoElements2svg",n),F("i2svg",n)):Promise.reject("Operation requires a DOM of some kind.")},watch:function(){var n=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},a=n.autoReplaceSvgRoot;d.autoReplaceSvg===!1&&(d.autoReplaceSvg=!0),d.observeMutations=!0,Me(function(){Ke({autoReplaceSvgRoot:a}),X("watch",n)})}},qe={icon:function(n){if(n===null)return null;if(Ot(n)==="object"&&n.prefix&&n.iconName)return{prefix:n.prefix,iconName:V(n.prefix,n.iconName)||n.iconName};if(Array.isArray(n)&&n.length===2){var a=n[1].indexOf("fa-")===0?n[1].slice(3):n[1],e=Nt(n[0]);return{prefix:e,iconName:V(e,a)||a}}if(typeof n=="string"&&(n.indexOf("".concat(d.cssPrefix,"-"))>-1||n.match(ge))){var r=Tt(n.split(" "),{skipLookups:!0});return{prefix:r.prefix||U(),iconName:V(r.prefix,r.iconName)||r.iconName}}if(typeof n=="string"){var i=U();return{prefix:i,iconName:V(i,n)||n}}}},I={noAuto:Xe,config:d,dom:Be,parse:qe,library:xa,findIconDefinition:qt,toHtml:ut},Ke=function(){var n=arguments.length>0&&arguments[0]!==void 0?arguments[0]:{},a=n.autoReplaceSvgRoot,e=a===void 0?x:a;(Object.keys(T.styles).length>0||d.autoFetchSvg)&&j&&d.autoReplaceSvg&&I.dom.i2svg({node:e})};function _t(t,n){return Object.defineProperty(t,"abstract",{get:n}),Object.defineProperty(t,"html",{get:function(){return t.abstract.map(function(e){return ut(e)})}}),Object.defineProperty(t,"node",{get:function(){if(j){var e=x.createElement("div");return e.innerHTML=t.html,e.children}}}),t}function Qe(t){var n=t.children,a=t.main,e=t.mask,r=t.attributes,i=t.styles,o=t.transform;if(sn(o)&&a.found&&!e.found){var s=a.width,f=a.height,c={x:s/f/2,y:.5};r.style=It(u(u({},i),{},{"transform-origin":"".concat(c.x+o.x/16,"em ").concat(c.y+o.y/16,"em")}))}return[{tag:"svg",attributes:r,children:n}]}function Ze(t){var n=t.prefix,a=t.iconName,e=t.children,r=t.attributes,i=t.symbol,o=i===!0?"".concat(n,"-").concat(d.cssPrefix,"-").concat(a):i;return[{tag:"svg",attributes:{style:"display: none;"},children:[{tag:"symbol",attributes:u(u({},r),{},{id:o}),children:e}]}]}function un(t){var n=t.icons,a=n.main,e=n.mask,r=t.prefix,i=t.iconName,o=t.transform,s=t.symbol,f=t.title,c=t.maskId,l=t.titleId,m=t.extra,g=t.watchable,h=g===void 0?!1:g,b=e.found?e:a,p=b.width,w=b.height,E=r==="fak",y=[d.replacementClass,i?"".concat(d.cssPrefix,"-").concat(i):""].filter(function(N){return m.classes.indexOf(N)===-1}).filter(function(N){return N!==""||!!N}).concat(m.classes).join(" "),S={children:[],attributes:u(u({},m.attributes),{},{"data-prefix":r,"data-icon":i,class:y,role:m.attributes.role||"img",xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 ".concat(p," ").concat(w)})},P=E&&!~m.classes.indexOf("fa-fw")?{width:"".concat(p/w*16*.0625,"em")}:{};h&&(S.attributes[G]=""),f&&(S.children.push({tag:"title",attributes:{id:S.attributes["aria-labelledby"]||"title-".concat(l||ft())},children:[f]}),delete S.attributes.title);var C=u(u({},S),{},{prefix:r,iconName:i,main:a,mask:e,maskId:c,transform:o,symbol:s,styles:u(u({},P),m.styles)}),_=e.found&&a.found?F("generateAbstractMask",C)||{children:[],attributes:{}}:F("generateAbstractIcon",C)||{children:[],attributes:{}},M=_.children,B=_.attributes;return C.children=M,C.attributes=B,s?Ze(C):Qe(C)}function Mn(t){var n=t.content,a=t.width,e=t.height,r=t.transform,i=t.title,o=t.extra,s=t.watchable,f=s===void 0?!1:s,c=u(u(u({},o.attributes),i?{title:i}:{}),{},{class:o.classes.join(" ")});f&&(c[G]="");var l=u({},o.styles);sn(r)&&(l.transform=Ie({transform:r,startCentered:!0,width:a,height:e}),l["-webkit-transform"]=l.transform);var m=It(l);m.length>0&&(c.style=m);var g=[];return g.push({tag:"span",attributes:c,children:[n]}),i&&g.push({tag:"span",attributes:{class:"sr-only"},children:[i]}),g}function Je(t){var n=t.content,a=t.title,e=t.extra,r=u(u(u({},e.attributes),a?{title:a}:{}),{},{class:e.classes.join(" ")}),i=It(e.styles);i.length>0&&(r.style=i);var o=[];return o.push({tag:"span",attributes:r,children:[n]}),a&&o.push({tag:"span",attributes:{class:"sr-only"},children:[a]}),o}var Ft=T.styles;function Kt(t){var n=t[0],a=t[1],e=t.slice(4),r=tn(e,1),i=r[0],o=null;return Array.isArray(i)?o={tag:"g",attributes:{class:"".concat(d.cssPrefix,"-").concat(W.GROUP)},children:[{tag:"path",attributes:{class:"".concat(d.cssPrefix,"-").concat(W.SECONDARY),fill:"currentColor",d:i[0]}},{tag:"path",attributes:{class:"".concat(d.cssPrefix,"-").concat(W.PRIMARY),fill:"currentColor",d:i[1]}}]}:o={tag:"path",attributes:{fill:"currentColor",d:i}},{found:!0,width:n,height:a,icon:o}}var $e={found:!1,width:512,height:512};function tr(t,n){!sa&&!d.showMissingIcons&&t&&console.error('Icon with name "'.concat(t,'" and prefix "').concat(n,'" is missing.'))}function Qt(t,n){var a=n;return n==="fa"&&d.styleDefault!==null&&(n=U()),new Promise(function(e,r){if(F("missingIconAbstract"),a==="fa"){var i=ka(t)||{};t=i.iconName||t,n=i.prefix||n}if(t&&n&&Ft[n]&&Ft[n][t]){var o=Ft[n][t];return e(Kt(o))}tr(t,n),e(u(u({},$e),{},{icon:d.showMissingIcons&&t?F("missingIconAbstract")||{}:{}}))})}var Ln=function(){},Zt=d.measurePerformance&&dt&&dt.mark&&dt.measure?dt:{mark:Ln,measure:Ln},nt='FA "6.5.2"',nr=function(n){return Zt.mark("".concat(nt," ").concat(n," begins")),function(){return wa(n)}},wa=function(n){Zt.mark("".concat(nt," ").concat(n," ends")),Zt.measure("".concat(nt," ").concat(n),"".concat(nt," ").concat(n," begins"),"".concat(nt," ").concat(n," ends"))},mn={begin:nr,end:wa},At=function(){};function zn(t){var n=t.getAttribute?t.getAttribute(G):null;return typeof n=="string"}function ar(t){var n=t.getAttribute?t.getAttribute(an):null,a=t.getAttribute?t.getAttribute(en):null;return n&&a}function er(t){return t&&t.classList&&t.classList.contains&&t.classList.contains(d.replacementClass)}function rr(){if(d.autoReplaceSvg===!0)return St.replace;var t=St[d.autoReplaceSvg];return t||St.replace}function ir(t){return x.createElementNS("http://www.w3.org/2000/svg",t)}function or(t){return x.createElement(t)}function Aa(t){var n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},a=n.ceFn,e=a===void 0?t.tag==="svg"?ir:or:a;if(typeof t=="string")return x.createTextNode(t);var r=e(t.tag);Object.keys(t.attributes||[]).forEach(function(o){r.setAttribute(o,t.attributes[o])});var i=t.children||[];return i.forEach(function(o){r.appendChild(Aa(o,{ceFn:e}))}),r}function sr(t){var n=" ".concat(t.outerHTML," ");return n="".concat(n,"Font Awesome fontawesome.com "),n}var St={replace:function(n){var a=n[0];if(a.parentNode)if(n[1].forEach(function(r){a.parentNode.insertBefore(Aa(r),a)}),a.getAttribute(G)===null&&d.keepOriginalSource){var e=x.createComment(sr(a));a.parentNode.replaceChild(e,a)}else a.remove()},nest:function(n){var a=n[0],e=n[1];if(~on(a).indexOf(d.replacementClass))return St.replace(n);var r=new RegExp("".concat(d.cssPrefix,"-.*"));if(delete e[0].attributes.id,e[0].attributes.class){var i=e[0].attributes.class.split(" ").reduce(function(s,f){return f===d.replacementClass||f.match(r)?s.toSvg.push(f):s.toNode.push(f),s},{toNode:[],toSvg:[]});e[0].attributes.class=i.toSvg.join(" "),i.toNode.length===0?a.removeAttribute("class"):a.setAttribute("class",i.toNode.join(" "))}var o=e.map(function(s){return ut(s)}).join(`
`);a.setAttribute(G,""),a.innerHTML=o}};function Rn(t){t()}function Sa(t,n){var a=typeof n=="function"?n:At;if(t.length===0)a();else{var e=Rn;d.mutateApproach===me&&(e=Y.requestAnimationFrame||Rn),e(function(){var r=rr(),i=mn.begin("mutate");t.map(r),i(),a()})}}var dn=!1;function Ea(){dn=!0}function Jt(){dn=!1}var Ct=null;function Fn(t){if(On&&d.observeMutations){var n=t.treeCallback,a=n===void 0?At:n,e=t.nodeCallback,r=e===void 0?At:e,i=t.pseudoElementsCallback,o=i===void 0?At:i,s=t.observeMutationsRoot,f=s===void 0?x:s;Ct=new On(function(c){if(!dn){var l=U();J(c).forEach(function(m){if(m.type==="childList"&&m.addedNodes.length>0&&!zn(m.addedNodes[0])&&(d.searchPseudoElements&&o(m.target),a(m.target)),m.type==="attributes"&&m.target.parentNode&&d.searchPseudoElements&&o(m.target.parentNode),m.type==="attributes"&&zn(m.target)&&~ye.indexOf(m.attributeName))if(m.attributeName==="class"&&ar(m.target)){var g=Tt(on(m.target)),h=g.prefix,b=g.iconName;m.target.setAttribute(an,h||l),b&&m.target.setAttribute(en,b)}else er(m.target)&&r(m.target)})}}),j&&Ct.observe(f,{childList:!0,attributes:!0,characterData:!0,subtree:!0})}}function fr(){Ct&&Ct.disconnect()}function lr(t){var n=t.getAttribute("style"),a=[];return n&&(a=n.split(";").reduce(function(e,r){var i=r.split(":"),o=i[0],s=i.slice(1);return o&&s.length>0&&(e[o]=s.join(":").trim()),e},{})),a}function cr(t){var n=t.getAttribute("data-prefix"),a=t.getAttribute("data-icon"),e=t.innerText!==void 0?t.innerText.trim():"",r=Tt(on(t));return r.prefix||(r.prefix=U()),n&&a&&(r.prefix=n,r.iconName=a),r.iconName&&r.prefix||(r.prefix&&e.length>0&&(r.iconName=Ue(r.prefix,t.innerText)||ln(r.prefix,Gt(t.innerText))),!r.iconName&&d.autoFetchSvg&&t.firstChild&&t.firstChild.nodeType===Node.TEXT_NODE&&(r.iconName=t.firstChild.data)),r}function ur(t){var n=J(t.attributes).reduce(function(r,i){return r.name!=="class"&&r.name!=="style"&&(r[i.name]=i.value),r},{}),a=t.getAttribute("title"),e=t.getAttribute("data-fa-title-id");return d.autoA11y&&(a?n["aria-labelledby"]="".concat(d.replacementClass,"-title-").concat(e||ft()):(n["aria-hidden"]="true",n.focusable="false")),n}function mr(){return{iconName:null,title:null,titleId:null,prefix:null,transform:L,symbol:!1,mask:{iconName:null,prefix:null,rest:[]},maskId:null,extra:{classes:[],styles:{},attributes:{}}}}function jn(t){var n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{styleParser:!0},a=cr(t),e=a.iconName,r=a.prefix,i=a.rest,o=ur(t),s=Bt("parseNodeAttributes",{},t),f=n.styleParser?lr(t):[];return u({iconName:e,title:t.getAttribute("title"),titleId:t.getAttribute("data-fa-title-id"),prefix:r,transform:L,mask:{iconName:null,prefix:null,rest:[]},maskId:null,symbol:!1,extra:{classes:i,styles:f,attributes:o}},s)}var dr=T.styles;function Oa(t){var n=d.autoReplaceSvg==="nest"?jn(t,{styleParser:!1}):jn(t);return~n.extra.classes.indexOf(fa)?F("generateLayersText",t,n):F("generateSvgReplacementMutation",t,n)}var H=new Set;rn.map(function(t){H.add("fa-".concat(t))});Object.keys(rt[k]).map(H.add.bind(H));Object.keys(rt[A]).map(H.add.bind(H));H=lt(H);function Dn(t){var n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:null;if(!j)return Promise.resolve();var a=x.documentElement.classList,e=function(m){return a.add("".concat(Pn,"-").concat(m))},r=function(m){return a.remove("".concat(Pn,"-").concat(m))},i=d.autoFetchSvg?H:rn.map(function(l){return"fa-".concat(l)}).concat(Object.keys(dr));i.includes("fa")||i.push("fa");var o=[".".concat(fa,":not([").concat(G,"])")].concat(i.map(function(l){return".".concat(l,":not([").concat(G,"])")})).join(", ");if(o.length===0)return Promise.resolve();var s=[];try{s=J(t.querySelectorAll(o))}catch{}if(s.length>0)e("pending"),r("complete");else return Promise.resolve();var f=mn.begin("onTree"),c=s.reduce(function(l,m){try{var g=Oa(m);g&&l.push(g)}catch(h){sa||h.name==="MissingIcon"&&console.error(h)}return l},[]);return new Promise(function(l,m){Promise.all(c).then(function(g){Sa(g,function(){e("active"),e("complete"),r("pending"),typeof n=="function"&&n(),f(),l()})}).catch(function(g){f(),m(g)})})}function vr(t){var n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:null;Oa(t).then(function(a){a&&Sa([a],n)})}function gr(t){return function(n){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},e=(n||{}).icon?n:qt(n||{}),r=a.mask;return r&&(r=(r||{}).icon?r:qt(r||{})),t(e,u(u({},a),{},{mask:r}))}}var br=function(n){var a=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},e=a.transform,r=e===void 0?L:e,i=a.symbol,o=i===void 0?!1:i,s=a.mask,f=s===void 0?null:s,c=a.maskId,l=c===void 0?null:c,m=a.title,g=m===void 0?null:m,h=a.titleId,b=h===void 0?null:h,p=a.classes,w=p===void 0?[]:p,E=a.attributes,y=E===void 0?{}:E,S=a.styles,P=S===void 0?{}:S;if(n){var C=n.prefix,_=n.iconName,M=n.icon;return _t(u({type:"icon"},n),function(){return X("beforeDOMElementCreation",{iconDefinition:n,params:a}),d.autoA11y&&(g?y["aria-labelledby"]="".concat(d.replacementClass,"-title-").concat(b||ft()):(y["aria-hidden"]="true",y.focusable="false")),un({icons:{main:Kt(M),mask:f?Kt(f.icon):{found:!1,width:null,height:null,icon:{}}},prefix:C,iconName:_,transform:u(u({},L),r),symbol:o,title:g,maskId:l,titleId:b,extra:{attributes:y,styles:P,classes:w}})})}},hr={mixout:function(){return{icon:gr(br)}},hooks:function(){return{mutationObserverCallbacks:function(a){return a.treeCallback=Dn,a.nodeCallback=vr,a}}},provides:function(n){n.i2svg=function(a){var e=a.node,r=e===void 0?x:e,i=a.callback,o=i===void 0?function(){}:i;return Dn(r,o)},n.generateSvgReplacementMutation=function(a,e){var r=e.iconName,i=e.title,o=e.titleId,s=e.prefix,f=e.transform,c=e.symbol,l=e.mask,m=e.maskId,g=e.extra;return new Promise(function(h,b){Promise.all([Qt(r,s),l.iconName?Qt(l.iconName,l.prefix):Promise.resolve({found:!1,width:512,height:512,icon:{}})]).then(function(p){var w=tn(p,2),E=w[0],y=w[1];h([a,un({icons:{main:E,mask:y},prefix:s,iconName:r,transform:f,symbol:c,maskId:m,title:i,titleId:o,extra:g,watchable:!0})])}).catch(b)})},n.generateAbstractIcon=function(a){var e=a.children,r=a.attributes,i=a.main,o=a.transform,s=a.styles,f=It(s);f.length>0&&(r.style=f);var c;return sn(o)&&(c=F("generateAbstractTransformGrouping",{main:i,transform:o,containerWidth:i.width,iconWidth:i.width})),e.push(c||i.icon),{children:e,attributes:r}}}},pr={mixout:function(){return{layer:function(a){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=e.classes,i=r===void 0?[]:r;return _t({type:"layer"},function(){X("beforeDOMElementCreation",{assembler:a,params:e});var o=[];return a(function(s){Array.isArray(s)?s.map(function(f){o=o.concat(f.abstract)}):o=o.concat(s.abstract)}),[{tag:"span",attributes:{class:["".concat(d.cssPrefix,"-layers")].concat(lt(i)).join(" ")},children:o}]})}}}},yr={mixout:function(){return{counter:function(a){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=e.title,i=r===void 0?null:r,o=e.classes,s=o===void 0?[]:o,f=e.attributes,c=f===void 0?{}:f,l=e.styles,m=l===void 0?{}:l;return _t({type:"counter",content:a},function(){return X("beforeDOMElementCreation",{content:a,params:e}),Je({content:a.toString(),title:i,extra:{attributes:c,styles:m,classes:["".concat(d.cssPrefix,"-layers-counter")].concat(lt(s))}})})}}}},kr={mixout:function(){return{text:function(a){var e=arguments.length>1&&arguments[1]!==void 0?arguments[1]:{},r=e.transform,i=r===void 0?L:r,o=e.title,s=o===void 0?null:o,f=e.classes,c=f===void 0?[]:f,l=e.attributes,m=l===void 0?{}:l,g=e.styles,h=g===void 0?{}:g;return _t({type:"text",content:a},function(){return X("beforeDOMElementCreation",{content:a,params:e}),Mn({content:a,transform:u(u({},L),i),title:s,extra:{attributes:m,styles:h,classes:["".concat(d.cssPrefix,"-layers-text")].concat(lt(c))}})})}}},provides:function(n){n.generateLayersText=function(a,e){var r=e.title,i=e.transform,o=e.extra,s=null,f=null;if(ra){var c=parseInt(getComputedStyle(a).fontSize,10),l=a.getBoundingClientRect();s=l.width/c,f=l.height/c}return d.autoA11y&&!r&&(o.attributes["aria-hidden"]="true"),Promise.resolve([a,Mn({content:a.innerHTML,width:s,height:f,transform:i,title:r,extra:o,watchable:!0})])}}},xr=new RegExp('"',"ug"),Yn=[1105920,1112319];function wr(t){var n=t.replace(xr,""),a=ze(n,0),e=a>=Yn[0]&&a<=Yn[1],r=n.length===2?n[0]===n[1]:!1;return{value:Gt(r?n[0]:n),isSecondary:e||r}}function Un(t,n){var a="".concat(ue).concat(n.replace(":","-"));return new Promise(function(e,r){if(t.getAttribute(a)!==null)return e();var i=J(t.children),o=i.filter(function(M){return M.getAttribute(Vt)===n})[0],s=Y.getComputedStyle(t,n),f=s.getPropertyValue("font-family").match(be),c=s.getPropertyValue("font-weight"),l=s.getPropertyValue("content");if(o&&!f)return t.removeChild(o),e();if(f&&l!=="none"&&l!==""){var m=s.getPropertyValue("content"),g=~["Sharp"].indexOf(f[2])?A:k,h=~["Solid","Regular","Light","Thin","Duotone","Brands","Kit"].indexOf(f[2])?it[g][f[2].toLowerCase()]:he[g][c],b=wr(m),p=b.value,w=b.isSecondary,E=f[0].startsWith("FontAwesome"),y=ln(h,p),S=y;if(E){var P=He(p);P.iconName&&P.prefix&&(y=P.iconName,h=P.prefix)}if(y&&!w&&(!o||o.getAttribute(an)!==h||o.getAttribute(en)!==S)){t.setAttribute(a,S),o&&t.removeChild(o);var C=mr(),_=C.extra;_.attributes[Vt]=n,Qt(y,h).then(function(M){var B=un(u(u({},C),{},{icons:{main:M,mask:cn()},prefix:h,iconName:S,extra:_,watchable:!0})),N=x.createElementNS("http://www.w3.org/2000/svg","svg");n==="::before"?t.insertBefore(N,t.firstChild):t.appendChild(N),N.outerHTML=B.map(function($){return ut($)}).join(`
`),t.removeAttribute(a),e()}).catch(r)}else e()}else e()})}function Ar(t){return Promise.all([Un(t,"::before"),Un(t,"::after")])}function Sr(t){return t.parentNode!==document.head&&!~de.indexOf(t.tagName.toUpperCase())&&!t.getAttribute(Vt)&&(!t.parentNode||t.parentNode.tagName!=="svg")}function Hn(t){if(j)return new Promise(function(n,a){var e=J(t.querySelectorAll("*")).filter(Sr).map(Ar),r=mn.begin("searchPseudoElements");Ea(),Promise.all(e).then(function(){r(),Jt(),n()}).catch(function(){r(),Jt(),a()})})}var Er={hooks:function(){return{mutationObserverCallbacks:function(a){return a.pseudoElementsCallback=Hn,a}}},provides:function(n){n.pseudoElements2svg=function(a){var e=a.node,r=e===void 0?x:e;d.searchPseudoElements&&Hn(r)}}},Wn=!1,Or={mixout:function(){return{dom:{unwatch:function(){Ea(),Wn=!0}}}},hooks:function(){return{bootstrap:function(){Fn(Bt("mutationObserverCallbacks",{}))},noAuto:function(){fr()},watch:function(a){var e=a.observeMutationsRoot;Wn?Jt():Fn(Bt("mutationObserverCallbacks",{observeMutationsRoot:e}))}}}},Vn=function(n){var a={size:16,x:0,y:0,flipX:!1,flipY:!1,rotate:0};return n.toLowerCase().split(" ").reduce(function(e,r){var i=r.toLowerCase().split("-"),o=i[0],s=i.slice(1).join("-");if(o&&s==="h")return e.flipX=!0,e;if(o&&s==="v")return e.flipY=!0,e;if(s=parseFloat(s),isNaN(s))return e;switch(o){case"grow":e.size=e.size+s;break;case"shrink":e.size=e.size-s;break;case"left":e.x=e.x-s;break;case"right":e.x=e.x+s;break;case"up":e.y=e.y-s;break;case"down":e.y=e.y+s;break;case"rotate":e.rotate=e.rotate+s;break}return e},a)},Pr={mixout:function(){return{parse:{transform:function(a){return Vn(a)}}}},hooks:function(){return{parseNodeAttributes:function(a,e){var r=e.getAttribute("data-fa-transform");return r&&(a.transform=Vn(r)),a}}},provides:function(n){n.generateAbstractTransformGrouping=function(a){var e=a.main,r=a.transform,i=a.containerWidth,o=a.iconWidth,s={transform:"translate(".concat(i/2," 256)")},f="translate(".concat(r.x*32,", ").concat(r.y*32,") "),c="scale(".concat(r.size/16*(r.flipX?-1:1),", ").concat(r.size/16*(r.flipY?-1:1),") "),l="rotate(".concat(r.rotate," 0 0)"),m={transform:"".concat(f," ").concat(c," ").concat(l)},g={transform:"translate(".concat(o/2*-1," -256)")},h={outer:s,inner:m,path:g};return{tag:"g",attributes:u({},h.outer),children:[{tag:"g",attributes:u({},h.inner),children:[{tag:e.icon.tag,children:e.icon.children,attributes:u(u({},e.icon.attributes),h.path)}]}]}}}},jt={x:0,y:0,width:"100%",height:"100%"};function Gn(t){var n=arguments.length>1&&arguments[1]!==void 0?arguments[1]:!0;return t.attributes&&(t.attributes.fill||n)&&(t.attributes.fill="black"),t}function Cr(t){return t.tag==="g"?t.children:[t]}var Ir={hooks:function(){return{parseNodeAttributes:function(a,e){var r=e.getAttribute("data-fa-mask"),i=r?Tt(r.split(" ").map(function(o){return o.trim()})):cn();return i.prefix||(i.prefix=U()),a.mask=i,a.maskId=e.getAttribute("data-fa-mask-id"),a}}},provides:function(n){n.generateAbstractMask=function(a){var e=a.children,r=a.attributes,i=a.main,o=a.mask,s=a.maskId,f=a.transform,c=i.width,l=i.icon,m=o.width,g=o.icon,h=Ce({transform:f,containerWidth:m,iconWidth:c}),b={tag:"rect",attributes:u(u({},jt),{},{fill:"white"})},p=l.children?{children:l.children.map(Gn)}:{},w={tag:"g",attributes:u({},h.inner),children:[Gn(u({tag:l.tag,attributes:u(u({},l.attributes),h.path)},p))]},E={tag:"g",attributes:u({},h.outer),children:[w]},y="mask-".concat(s||ft()),S="clip-".concat(s||ft()),P={tag:"mask",attributes:u(u({},jt),{},{id:y,maskUnits:"userSpaceOnUse",maskContentUnits:"userSpaceOnUse"}),children:[b,E]},C={tag:"defs",children:[{tag:"clipPath",attributes:{id:S},children:Cr(g)},P]};return e.push(C,{tag:"rect",attributes:u({fill:"currentColor","clip-path":"url(#".concat(S,")"),mask:"url(#".concat(y,")")},jt)}),{children:e,attributes:r}}}},Nr={provides:function(n){var a=!1;Y.matchMedia&&(a=Y.matchMedia("(prefers-reduced-motion: reduce)").matches),n.missingIconAbstract=function(){var e=[],r={fill:"currentColor"},i={attributeType:"XML",repeatCount:"indefinite",dur:"2s"};e.push({tag:"path",attributes:u(u({},r),{},{d:"M156.5,447.7l-12.6,29.5c-18.7-9.5-35.9-21.2-51.5-34.9l22.7-22.7C127.6,430.5,141.5,440,156.5,447.7z M40.6,272H8.5 c1.4,21.2,5.4,41.7,11.7,61.1L50,321.2C45.1,305.5,41.8,289,40.6,272z M40.6,240c1.4-18.8,5.2-37,11.1-54.1l-29.5-12.6 C14.7,194.3,10,216.7,8.5,240H40.6z M64.3,156.5c7.8-14.9,17.2-28.8,28.1-41.5L69.7,92.3c-13.7,15.6-25.5,32.8-34.9,51.5 L64.3,156.5z M397,419.6c-13.9,12-29.4,22.3-46.1,30.4l11.9,29.8c20.7-9.9,39.8-22.6,56.9-37.6L397,419.6z M115,92.4 c13.9-12,29.4-22.3,46.1-30.4l-11.9-29.8c-20.7,9.9-39.8,22.6-56.8,37.6L115,92.4z M447.7,355.5c-7.8,14.9-17.2,28.8-28.1,41.5 l22.7,22.7c13.7-15.6,25.5-32.9,34.9-51.5L447.7,355.5z M471.4,272c-1.4,18.8-5.2,37-11.1,54.1l29.5,12.6 c7.5-21.1,12.2-43.5,13.6-66.8H471.4z M321.2,462c-15.7,5-32.2,8.2-49.2,9.4v32.1c21.2-1.4,41.7-5.4,61.1-11.7L321.2,462z M240,471.4c-18.8-1.4-37-5.2-54.1-11.1l-12.6,29.5c21.1,7.5,43.5,12.2,66.8,13.6V471.4z M462,190.8c5,15.7,8.2,32.2,9.4,49.2h32.1 c-1.4-21.2-5.4-41.7-11.7-61.1L462,190.8z M92.4,397c-12-13.9-22.3-29.4-30.4-46.1l-29.8,11.9c9.9,20.7,22.6,39.8,37.6,56.9 L92.4,397z M272,40.6c18.8,1.4,36.9,5.2,54.1,11.1l12.6-29.5C317.7,14.7,295.3,10,272,8.5V40.6z M190.8,50 c15.7-5,32.2-8.2,49.2-9.4V8.5c-21.2,1.4-41.7,5.4-61.1,11.7L190.8,50z M442.3,92.3L419.6,115c12,13.9,22.3,29.4,30.5,46.1 l29.8-11.9C470,128.5,457.3,109.4,442.3,92.3z M397,92.4l22.7-22.7c-15.6-13.7-32.8-25.5-51.5-34.9l-12.6,29.5 C370.4,72.1,384.4,81.5,397,92.4z"})});var o=u(u({},i),{},{attributeName:"opacity"}),s={tag:"circle",attributes:u(u({},r),{},{cx:"256",cy:"364",r:"28"}),children:[]};return a||s.children.push({tag:"animate",attributes:u(u({},i),{},{attributeName:"r",values:"28;14;28;28;14;28;"})},{tag:"animate",attributes:u(u({},o),{},{values:"1;0;1;1;0;1;"})}),e.push(s),e.push({tag:"path",attributes:u(u({},r),{},{opacity:"1",d:"M263.7,312h-16c-6.6,0-12-5.4-12-12c0-71,77.4-63.9,77.4-107.8c0-20-17.8-40.2-57.4-40.2c-29.1,0-44.3,9.6-59.2,28.7 c-3.9,5-11.1,6-16.2,2.4l-13.1-9.2c-5.6-3.9-6.9-11.8-2.6-17.2c21.2-27.2,46.4-44.7,91.2-44.7c52.3,0,97.4,29.8,97.4,80.2 c0,67.6-77.4,63.5-77.4,107.8C275.7,306.6,270.3,312,263.7,312z"}),children:a?[]:[{tag:"animate",attributes:u(u({},o),{},{values:"1;0;0;0;0;1;"})}]}),a||e.push({tag:"path",attributes:u(u({},r),{},{opacity:"0",d:"M232.5,134.5l7,168c0.3,6.4,5.6,11.5,12,11.5h9c6.4,0,11.7-5.1,12-11.5l7-168c0.3-6.8-5.2-12.5-12-12.5h-23 C237.7,122,232.2,127.7,232.5,134.5z"}),children:[{tag:"animate",attributes:u(u({},o),{},{values:"0;0;1;1;0;0;"})}]}),{tag:"g",attributes:{class:"missing"},children:e}}}},Tr={hooks:function(){return{parseNodeAttributes:function(a,e){var r=e.getAttribute("data-fa-symbol"),i=r===null?!1:r===""?!0:r;return a.symbol=i,a}}}},_r=[Te,hr,pr,yr,kr,Er,Or,Pr,Ir,Nr,Tr];Ge(_r,{mixoutsTo:I});I.noAuto;I.config;var Vr=I.library,Gr=I.dom,$t=I.parse;I.findIconDefinition;I.toHtml;var Mr=I.icon;I.layer;I.text;I.counter;let Pa=!1;try{Pa=!0}catch{}function Lr(...t){!Pa&&console&&typeof console.error=="function"&&console.error(...t)}function Xn(t){if(t&&typeof t=="object"&&t.prefix&&t.iconName&&t.icon)return t;if($t.icon)return $t.icon(t);if(t===null)return null;if(t&&typeof t=="object"&&t.prefix&&t.iconName)return t;if(Array.isArray(t)&&t.length===2)return{prefix:t[0],iconName:t[1]};if(typeof t=="string")return{prefix:"fas",iconName:t}}function Dt(t,n){return Array.isArray(n)&&n.length>0||!Array.isArray(n)&&n?{[t]:n}:{}}function zr(t){let n,a,e=[t[2]],r={};for(let i=0;i<e.length;i+=1)r=Et(r,e[i]);return{c(){n=Ra("svg"),a=new Fa(!0),this.h()},l(i){n=ja(i,"svg",{});var o=Da(n);a=Ya(o,!0),o.forEach(Yt),this.h()},h(){a.a=null,Ua(n,r)},m(i,o){Jn(i,n,o),a.m(t[1],n),t[7](n)},p:Lt,i:Lt,o:Lt,d(i){i&&Yt(n),t[7](null)}}}function Rr(t,n,a){let{tag:e}=n,{props:r}=n,{children:i}=n,{style:o=null}=n,{ref:s=null}=n;if(e!=="svg")throw new Error('SvgElement requires a tag of "svg"');function f(b){return(b==null?void 0:b.reduce((p,w)=>p+(w.tag?c(w):w),""))||""}function c({tag:b,props:p,children:w}){const E=Object.keys(p).map(y=>`${y}="${p[y]}"`).join(" ");return`<${b} ${E}>${f(w)}</${b}>`}const l=f(i),m=r!=null&&r.style?`${r.style}${o||""}`:o,g={...r,style:m};function h(b){Kn[b?"unshift":"push"](()=>{s=b,a(0,s)})}return t.$$set=b=>{"tag"in b&&a(3,e=b.tag),"props"in b&&a(4,r=b.props),"children"in b&&a(5,i=b.children),"style"in b&&a(6,o=b.style),"ref"in b&&a(0,s=b.ref)},[s,l,g,e,r,i,o,h]}class Fr extends Qn{constructor(n){super(),Zn(this,n,Rr,zr,qn,{tag:3,props:4,children:5,style:6,ref:0})}}function Bn(t){let n,a,e;const r=[t[2],{style:t[1]}];function i(s){t[28](s)}let o={};for(let s=0;s<r.length;s+=1)o=Et(o,r[s]);return t[0]!==void 0&&(o.ref=t[0]),n=new Fr({props:o}),Kn.push(()=>Va(n,"ref",i)),{c(){Ga(n.$$.fragment)},l(s){Xa(n.$$.fragment,s)},m(s,f){Ba(n,s,f),e=!0},p(s,f){const c=f[0]&6?Ka(r,[f[0]&4&&Qa(s[2]),f[0]&2&&{style:s[1]}]):{};!a&&f[0]&1&&(a=!0,c.ref=s[0],za(()=>a=!1)),n.$set(c)},i(s){e||(wt(n.$$.fragment,s),e=!0)},o(s){Ut(n.$$.fragment,s),e=!1},d(s){qa(n,s)}}}function jr(t){let n,a,e=t[2]&&Bn(t);return{c(){e&&e.c(),n=xn()},l(r){e&&e.l(r),n=xn()},m(r,i){e&&e.m(r,i),Jn(r,n,i),a=!0},p(r,i){r[2]?e?(e.p(r,i),i[0]&4&&wt(e,1)):(e=Bn(r),e.c(),wt(e,1),e.m(n.parentNode,n)):e&&(Ha(),Ut(e,1,1,()=>{e=null}),Wa())},i(r){a||(wt(e),a=!0)},o(r){Ut(e),a=!1},d(r){r&&Yt(n),e&&e.d(r)}}}function Dr(t,n,a){const e=["border","mask","maskId","fixedWidth","inverse","flip","icon","listItem","pull","pulse","rotation","size","spin","spinPulse","spinReverse","beat","fade","beatFade","bounce","shake","symbol","title","titleId","transform","swapOpacity","ref","style"];let r=yn(n,e),{border:i=!1}=n,{mask:o=null}=n,{maskId:s=null}=n,{fixedWidth:f=!1}=n,{inverse:c=!1}=n,{flip:l=!1}=n,{icon:m=null}=n,{listItem:g=!1}=n,{pull:h=null}=n,{pulse:b=!1}=n,{rotation:p=null}=n,{size:w=null}=n,{spin:E=!1}=n,{spinPulse:y=!1}=n,{spinReverse:S=!1}=n,{beat:P=!1}=n,{fade:C=!1}=n,{beatFade:_=!1}=n,{bounce:M=!1}=n,{shake:B=!1}=n,{symbol:N=!1}=n,{title:$=""}=n,{titleId:Mt=null}=n,{transform:tt=null}=n,{swapOpacity:vn=!1}=n,{ref:mt=null}=n,{style:gn=null}=n;const bn=Xn(m),Ca=Dt("classes",[...Za(n),...(n.class||"").split(" ")]),Ia=Dt("transform",typeof tt=="string"?$t.transform(tt):tt),Na=Dt("mask",Xn(o)),hn=Mr(bn,{...Ca,...Ia,...Na,symbol:N,title:$,titleId:Mt,maskId:s});let pn=null;if(!hn)Lr("Could not find icon",bn);else{const{abstract:v}=hn;pn=$n((_a,Ma,La)=>({tag:_a,props:Ma,children:La}),v[0],r)}function Ta(v){mt=v,a(0,mt)}return t.$$set=v=>{a(35,n=Et(Et({},n),kn(v))),a(34,r=yn(n,e)),"border"in v&&a(3,i=v.border),"mask"in v&&a(4,o=v.mask),"maskId"in v&&a(5,s=v.maskId),"fixedWidth"in v&&a(6,f=v.fixedWidth),"inverse"in v&&a(7,c=v.inverse),"flip"in v&&a(8,l=v.flip),"icon"in v&&a(9,m=v.icon),"listItem"in v&&a(10,g=v.listItem),"pull"in v&&a(11,h=v.pull),"pulse"in v&&a(12,b=v.pulse),"rotation"in v&&a(13,p=v.rotation),"size"in v&&a(14,w=v.size),"spin"in v&&a(15,E=v.spin),"spinPulse"in v&&a(16,y=v.spinPulse),"spinReverse"in v&&a(17,S=v.spinReverse),"beat"in v&&a(18,P=v.beat),"fade"in v&&a(19,C=v.fade),"beatFade"in v&&a(20,_=v.beatFade),"bounce"in v&&a(21,M=v.bounce),"shake"in v&&a(22,B=v.shake),"symbol"in v&&a(23,N=v.symbol),"title"in v&&a(24,$=v.title),"titleId"in v&&a(25,Mt=v.titleId),"transform"in v&&a(26,tt=v.transform),"swapOpacity"in v&&a(27,vn=v.swapOpacity),"ref"in v&&a(0,mt=v.ref),"style"in v&&a(1,gn=v.style)},n=kn(n),[mt,gn,pn,i,o,s,f,c,l,m,g,h,b,p,w,E,y,S,P,C,_,M,B,N,$,Mt,tt,vn,Ta]}class Xr extends Qn{constructor(n){super(),Zn(this,n,Dr,jr,qn,{border:3,mask:4,maskId:5,fixedWidth:6,inverse:7,flip:8,icon:9,listItem:10,pull:11,pulse:12,rotation:13,size:14,spin:15,spinPulse:16,spinReverse:17,beat:18,fade:19,beatFade:20,bounce:21,shake:22,symbol:23,title:24,titleId:25,transform:26,swapOpacity:27,ref:0,style:1},null,[-1,-1])}}var Yr={prefix:"fas",iconName:"square-up-right",icon:[448,512,[8599,"external-link-square-alt"],"f360","M384 32c35.3 0 64 28.7 64 64V416c0 35.3-28.7 64-64 64H64c-35.3 0-64-28.7-64-64V96C0 60.7 28.7 32 64 32H384zM320 313.4V176c0-8.8-7.2-16-16-16H166.6c-12.5 0-22.6 10.1-22.6 22.6c0 6 2.4 11.8 6.6 16L184 232l-66.3 66.3C114 302 112 306.9 112 312s2 10 5.7 13.7l36.7 36.7c3.6 3.6 8.5 5.7 13.7 5.7s10-2 13.7-5.7L248 296l33.4 33.4c4.2 4.2 10 6.6 16 6.6c12.5 0 22.6-10.1 22.6-22.6z"]},Br=Yr,qr={prefix:"fas",iconName:"arrows-rotate",icon:[512,512,[128472,"refresh","sync"],"f021","M105.1 202.6c7.7-21.8 20.2-42.3 37.8-59.8c62.5-62.5 163.8-62.5 226.3 0L386.3 160H352c-17.7 0-32 14.3-32 32s14.3 32 32 32H463.5c0 0 0 0 0 0h.4c17.7 0 32-14.3 32-32V80c0-17.7-14.3-32-32-32s-32 14.3-32 32v35.2L414.4 97.6c-87.5-87.5-229.3-87.5-316.8 0C73.2 122 55.6 150.7 44.8 181.4c-5.9 16.7 2.9 34.9 19.5 40.8s34.9-2.9 40.8-19.5zM39 289.3c-5 1.5-9.8 4.2-13.7 8.2c-4 4-6.7 8.8-8.1 14c-.3 1.2-.6 2.5-.8 3.8c-.3 1.7-.4 3.4-.4 5.1V432c0 17.7 14.3 32 32 32s32-14.3 32-32V396.9l17.6 17.5 0 0c87.5 87.4 229.3 87.4 316.7 0c24.4-24.4 42.1-53.1 52.9-83.7c5.9-16.7-2.9-34.9-19.5-40.8s-34.9 2.9-40.8 19.5c-7.7 21.8-20.2 42.3-37.8 59.8c-62.5 62.5-163.8 62.5-226.3 0l-.1-.1L125.6 352H160c17.7 0 32-14.3 32-32s-14.3-32-32-32H48.4c-1.6 0-3.2 .1-4.8 .3s-3.1 .5-4.6 1z"]},Ur={prefix:"fas",iconName:"circle-info",icon:[512,512,["info-circle"],"f05a","M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216 336h24V272H216c-13.3 0-24-10.7-24-24s10.7-24 24-24h48c13.3 0 24 10.7 24 24v88h8c13.3 0 24 10.7 24 24s-10.7 24-24 24H216c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32 32 0 1 1 0 64 32 32 0 1 1 0-64z"]},Kr=Ur,Qr={prefix:"fas",iconName:"forward",icon:[512,512,[9193],"f04e","M52.5 440.6c-9.5 7.9-22.8 9.7-34.1 4.4S0 428.4 0 416V96C0 83.6 7.2 72.3 18.4 67s24.5-3.6 34.1 4.4L224 214.3V256v41.7L52.5 440.6zM256 352V256 128 96c0-12.4 7.2-23.7 18.4-29s24.5-3.6 34.1 4.4l192 160c7.3 6.1 11.5 15.1 11.5 24.6s-4.2 18.5-11.5 24.6l-192 160c-9.5 7.9-22.8 9.7-34.1 4.4s-18.4-16.6-18.4-29V352z"]};export{Xr as F,Kr as a,qr as b,Qr as c,Gr as d,Br as f,Vr as l};
