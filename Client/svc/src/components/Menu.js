import React, {useState} from "react";
import { Box, Text, Link } from "gestalt";
import "gestalt/dist/gestalt.css";
import "bootstrap/dist/css/bootstrap.css";

const Menu = () => {
  const [menuIdx, setMenuInx] = useState(0)

  const onHandleChange = (idx) => {
    setMenuInx(idx)
  }
  
  const memuItems = () => {
    const arr = [
      {link:'#', name:'About SVC'},
      {link:'#', name:'Skills'},
      {link:'#', name:'Projects'}
    ];

    return arr.map((obj,idx) => {
      const active = (idx === menuIdx?true:false)
      return (
        <Text color="white" size={active?'lg':'md'} bold={active}>
          <Link href={obj.link} onClick={() => onHandleChange(idx)}>
            <Box padding={3}>{obj.name}</Box>
          </Link>
        </Text>
      )
    })
  }

  return (
    <Box display="flex" column={12} justifyContent="center" color="orchid">
      {memuItems()}
    </Box>
  )
}

export default Menu;